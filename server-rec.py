import asyncio
import os
import tempfile
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Body, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import whisper
import json 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient, DESCENDING
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import logging
import re
import requests
import aiohttp

# At the top of your file, configure logging (if not already done)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCHAPI_KEY = os.getenv("SEARCHAPI_KEY")  # Add this to your .env file

# Initialize FastAPI app
app = FastAPI(
    title="AI Expense Assistant API with Voice Transcription",
    description="API for querying expense data using natural language and voice",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React Native app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize whisper model
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        # Load whisper model
        model = whisper.load_model("base")
        print("Whisper model loaded successfully")
        
        # Create MongoDB indexes
        conversations_collection = client["klip"]["user_conversations"]
        conversations_collection.create_index([("user_id", 1), ("timestamp", -1)])
        print("MongoDB indexes created successfully")
    except Exception as e:
        print(f"Error during startup: {e}")

# Initialize embeddings and MongoDB connection
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    task_type="RETRIEVAL_QUERY"
)
client = MongoClient(MONGO_URI)
db = client["klip"]
collection = db["flattened_manual_expenses"]
conversations_collection = db["user_conversations_items"]

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="receipts_vector_index_dev"
)

# Retailer receipts vector store (add this)
retailer_collection = db["flattened_manual_expenses_retailer"]
retailer_vector_store = MongoDBAtlasVectorSearch(
    collection=retailer_collection,
    embedding=embeddings,
    index_name="retailer_vector_index_dev"
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# Define Pydantic models for request/response
class Message(BaseModel):
    role: str
    content: str

class ChatHistoryItem(BaseModel):
    role: str
    content: str
    
class QueryRequest(BaseModel):
    user_id: str
    query: str
    chat_history: Optional[List[ChatHistoryItem]] = Field(default_factory=list)

class SourceReceipt(BaseModel):
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class TranscriptionResponse(BaseModel):
    success: bool
    message: str
    text: Optional[str] = None

class VoiceQueryResponse(BaseModel):
    transcription: TranscriptionResponse
    query_response: Optional[QueryResponse] = None

# Google Shopping API Integration
async def search_google_shopping(query: str, filters: Dict[str, Any] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search Google Shopping API through SearchAPI for product recommendations
    """
    if not SEARCHAPI_KEY:
        logger.warning("SEARCHAPI_KEY not configured, skipping Google Shopping search")
        return []
    
    try:
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "google_shopping",
            "q": query,
            "location": "United Arab Emirates",
            "gl": "ae",
            "hl": "en",
            "google_domain": "google.ae",
            "api_key": SEARCHAPI_KEY,
            "num": min(limit, 100)  # Max 100 per API docs
        }
        
        # Apply filters if provided
        if filters:
            if filters.get("price_min"):
                params["price_min"] = filters["price_min"]
            if filters.get("price_max"):
                params["price_max"] = filters["price_max"]
            if filters.get("condition"):
                params["condition"] = filters["condition"]
            if filters.get("sort_by"):
                params["sort_by"] = filters["sort_by"]
            if filters.get("tbs"):
                params["tbs"] = filters["tbs"]
            if filters.get("page"):
                params["page"] = filters["page"]
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"Google Shopping API Response: {json.dumps(data, indent=2)}")
                    products = []
                    
                    # Extract shopping results
                    shopping_results = data.get("shopping_results", [])
                    for item in shopping_results[:limit]:
                        product = {
                            "title": item.get("title", ""),
                            "price": item.get("price", ""),
                            "source": item.get("source", ""),
                            "store": item.get("seller", ""),
                            "link": str(item.get("product_link", "")),
                            "thumbnail": item.get("thumbnail", ""),
                            "rating": item.get("rating", ""),
                            "reviews": item.get("reviews", ""),
                            "delivery": item.get("delivery", ""),
                            "condition": item.get("condition", ""),
                            "product_id": item.get("product_id", "")
                        }
                        products.append(product)
                    
                    logger.info(f"Found {len(products)} products from Google Shopping for query: {query} with filters: {filters}")
                    return products
                else:
                    logger.error(f"Google Shopping API error: {response.status}")
                    return []
                    
    except Exception as e:
        logger.error(f"Error searching Google Shopping: {str(e)}")
        return []

def extract_search_terms_and_filters_from_query(query: str) -> tuple[str, Dict[str, Any]]:
    """
    Extract relevant search terms and filters from the user query for Google Shopping
    Uses LLM to parse complex queries and extract filters
    """
    extraction_prompt = f"""
    Analyze this shopping query and extract:
    1. Main product search terms (just the product keywords)
    2. Price filters (if mentioned)
    3. Condition preferences (new/used)
    4. Sort preferences
    
    Return JSON format:
    {{
        "search_terms": "main product keywords only",
        "filters": {{
            "price_min": number or null,
            "price_max": number or null,
            "condition": "new" or "used" or null,
            "sort_by": "relevance" or "price_low_to_high" or "price_high_to_low" or "review_score" or null
        }}
    }}
    
    Examples:
    - "airpods under 300 AED" → {{"search_terms": "airpods", "filters": {{"price_max": 300}}}}
    - "cheap used laptops" → {{"search_terms": "laptops", "filters": {{"condition": "used", "sort_by": "price_low_to_high"}}}}
    - "best rated headphones" → {{"search_terms": "headphones", "filters": {{"sort_by": "review_score"}}}}
    - "gaming mouse between 50 and 200 AED" → {{"search_terms": "gaming mouse", "filters": {{"price_min": 50, "price_max": 200}}}}
    
    Query: {query}
    """
    
    try:
        response = llm.invoke(extraction_prompt)
        content = response.content.strip()
        
        # Clean markdown code blocks
        if content.startswith("```"):
            first_newline = content.find('\n')
            if first_newline != -1:
                content = content[first_newline + 1:]
            else:
                content = content[3:]
                if content.startswith("json"):
                    content = content[4:].strip()
        
        if content.endswith("```"):
            content = content[:-3].strip()
        
        content = content.strip()
        
        extraction_result = json.loads(content)
        search_terms = extraction_result.get("search_terms", query)
        filters = extraction_result.get("filters", {})
        
        # Clean up filters - remove null values
        clean_filters = {k: v for k, v in filters.items() if v is not None}
        
        logger.info(f"Extracted search terms: '{search_terms}', filters: {clean_filters}")
        return search_terms, clean_filters
        
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Error extracting search terms and filters: {str(e)}")
        # Fallback: return original query with no filters
        return query, {}

def format_shopping_results_for_llm(products: List[Dict[str, Any]]) -> str:
    """
    Format Google Shopping results for inclusion in LLM context
    """
    if not products:
        return "No shopping results found."
    
    formatted_results = "Shopping Results:\n\n"
    for i, product in enumerate(products, 1):
        formatted_results += f"{i}. {product.get('title', 'Unknown Product')}\n"
        formatted_results += f"   Price: {product.get('price', 'N/A')}\n"
        formatted_results += f"   Store: {product.get('store', 'N/A')}\n"
        if product.get('rating'):
            formatted_results += f"   Rating: {product.get('rating')} ({product.get('reviews', 'N/A')} reviews)\n"
        if product.get('delivery'):
            formatted_results += f"   Delivery: {product.get('delivery')}\n"
        if product.get('condition'):
            formatted_results += f"   Condition: {product.get('condition')}\n"
        formatted_results += f"   Link: {product.get('link', 'N/A')}\n\n"
    
    return formatted_results

# Conversation Manager class for handling chat history
class ConversationManager:
    def __init__(self):
        self.collection = conversations_collection
    
    async def save_message(self, user_id: str, role: str, content: str):
        """Save a new message to the conversation history"""
        message_doc = {
            "user_id": user_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        }
        result = self.collection.insert_one(message_doc)
        return result.inserted_id
    
    async def get_chat_history(self, user_id: str, max_messages: int = 10) -> List[dict]:
        """Retrieve recent chat history for a user"""
        cursor = self.collection.find(
            {"user_id": user_id}
        ).sort("timestamp", DESCENDING).limit(max_messages)
        
        # Convert to list and reverse to get chronological order
        messages = list(cursor)
        messages.reverse()
        
        return [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in messages
        ]
    
    async def clear_history(self, user_id: str):
        """Clear a user's conversation history"""
        result = self.collection.delete_many({"user_id": user_id})
        return result.deleted_count

# Factory for dependency injection
def get_conversation_manager():
    return ConversationManager()

# --- Intent classification prompt and function ---
INTENT_PROMPT = """Classify this query into one of two categories:
- "personal" (about user's own receipts)
- "recommendation" (seeking general recommendations)
Return JSON format: {{"intent": ""}}
Query: {query}"""

async def classify_query_intent(query: str) -> str:
    try:
        response = await llm.ainvoke(INTENT_PROMPT.format(query=query))
        content = response.content.strip()
        
        # More robust markdown code block removal
        if content.startswith("```"):
            # Find the first newline after ```
            first_newline = content.find('\n')
            if first_newline != -1:
                content = content[first_newline + 1:]
            else:
                # If no newline, remove ```json or just ```
                content = content[3:]
                if content.startswith("json"):
                    content = content[4:].strip()
        
        if content.endswith("```"):
            content = content[:-3].strip()
        
        # Additional cleanup - remove any remaining whitespace
        content = content.strip()
        
        logger.info(f"Cleaned content for JSON parsing: '{content}'")
        
        classification = json.loads(content)
        intent = classification.get("intent", "personal")
        
        # Validate the intent value
        if intent not in ["personal", "recommendation"]:
            logger.warning(f"Invalid intent value '{intent}', defaulting to 'personal'")
            intent = "personal"
            
        logger.info(f"Successfully classified intent: {intent}")
        return intent
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {str(e)}; cleaned content: '{content}'")
        return "personal"
    except Exception as e:
        logger.error(f"Intent classification failed: {str(e)}; raw response: {getattr(response, 'content', 'No content')}")
        return "personal"
    
# Create retrieval chain factory
# Modified retrieval chain factory
async def get_retrieval_chain(user_id: str, query: str):
    try:
        intent = await classify_query_intent(query)
        logger.info(f"Intent value: {intent}, type: {type(intent)}")
    except Exception as e:
        logger.error(f"Intent classification error: {str(e)}")
        intent = "personal"

    search_kwargs = {"k": 5}

    if intent == "personal":
        search_kwargs["pre_filter"] = {"metadata.user_id": user_id}
        logger.info(f"Using user vector store with filters: {search_kwargs}")
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    elif intent == "recommendation":
        logger.info(f"Using retailer vector store with filters: {search_kwargs}")
        retriever = retailer_vector_store.as_retriever(search_kwargs=search_kwargs)
    else:
        logger.info(f"Defaulting to user vector store with filters: {search_kwargs}")
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("human", "Given the conversation history, reformulate this as a standalone query about expenses:"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, retriever_prompt
    )

    # Answer generation chain
    if intent == "personal":
        system_prompt = """You are a smart expense assistant. Use the user's personal receipts and conversation history:

        Personal Receipts:
        {context}

        Conversation History:
        {chat_history}

        Answer questions about the user's own expenses, spending patterns, and personal purchase history. Use specific amounts, dates, and merchants from their receipts."""
    else:  # recommendation intent
        system_prompt = """You are a smart expense assistant. Use these retailer receipts and conversation history to provide recommendations:

        Retailer Database:
        {context}

        Conversation History:
        {chat_history}

        When providing recommendations, use the retailer data to suggest where users can buy items. Mention that "Another user of the app has purchased [item] at [retailer name] for [price]" and include location information when available along with date of purchase. Focus on real purchase data from the retailer database."""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain), intent

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Expense Assistant API with Voice Integration"}

@app.post("/query", response_model=QueryResponse)
async def query_expenses(
    request: QueryRequest,
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    # Validate user_id
    if not request.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    # Get chat history if not provided in request
    chat_history = request.chat_history
    if not chat_history:
        chat_history = await conversation_manager.get_chat_history(request.user_id)
    
    # Convert chat history to LangChain messages with proper validation
    lc_messages = []
    for msg in chat_history:
        # Check if msg is a dict (from get_chat_history) or a ChatHistoryItem
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
        
        # Strict validation - only add if both role and content are valid
        if role and content and content.strip():
            if role.lower() in ["user", "you", "human"]:
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))

    # Get retrieval chain for this user
    retrieval_chain, intent = await get_retrieval_chain(request.user_id, request.query)
    
    # Process query
    try:
        # Save the user's query to history
        await conversation_manager.save_message(request.user_id, "user", request.query)
        
        response = retrieval_chain.invoke({
            "input": request.query,
            "chat_history": lc_messages
        })
        
        # Check if this is a recommendation query and check if exact item is present
        shopping_results = []
        if intent == "recommendation":
            # Extract search terms for checking exact matches
            search_terms, filters = extract_search_terms_and_filters_from_query(request.query)
            
            # Check if the exact item is present in database results
            exact_item_found = False
            if len(response["context"]) > 0:
                search_terms_lower = search_terms.lower()
                search_words = search_terms_lower.split()
                
                for doc in response["context"]:
                    doc_content = doc.page_content.lower()
                    # Check if any of the search terms appear in the document
                    if any(word in doc_content for word in search_words):
                        exact_item_found = True
                        break
            
            # Use Google Shopping only if exact item is not found in database
            if not exact_item_found:
                logger.info(f"Exact item '{search_terms}' not found in retailer database, searching Google Shopping")
                shopping_results = await search_google_shopping(search_terms, filters)
                
                if shopping_results:
                    # Add shopping results to the context for the LLM
                    shopping_context = format_shopping_results_for_llm(shopping_results)
                    
                    # Create a new prompt that includes shopping results
                    enhanced_system_prompt = """You are a smart expense assistant. Use these retailer receipts, conversation history, and shopping results to provide recommendations:

                    Retailer Database:
                    {context}

                    Shopping Results:
                    {shopping_context}

                    Conversation History:
                    {chat_history}

                    The exact item was not found in our retailer database, so here are current shopping options. Include prices, ratings, store names, and product links from the shopping results. Be specific about where users can buy items online. Always mention the store name and include the product link in your response."""

                    enhanced_qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", enhanced_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])

                    enhanced_question_answer_chain = create_stuff_documents_chain(llm, enhanced_qa_prompt)
                    
                    # Re-invoke with shopping context
                    response = enhanced_question_answer_chain.invoke({
                        "input": request.query,
                        "context": response["context"],
                        "shopping_context": shopping_context,
                        "chat_history": lc_messages
                    })
                    
                    # Convert response format to match expected structure
                    if isinstance(response, str):
                        response = {
                            "answer": response,
                            "context": []
                        }
            else:
                logger.info(f"Exact item '{search_terms}' found in retailer database, using database results only")
        
        # Save the AI's response to history
        await conversation_manager.save_message(request.user_id, "assistant", response["answer"])
        
        # Format sources for the response
        sources = []
        for doc in response.get("context", []):
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Add shopping results as sources if they were used
        for product in shopping_results:
            sources.append({
                "content": f"Product: {product.get('title', '')} - Price: {product.get('price', '')}",
                "metadata": {
                    "source": "google_shopping",
                    "title": product.get("title", ""),
                    "price": product.get("price", ""),
                    "link": product.get("link", ""),
                    "rating": product.get("rating", ""),
                    "reviews": product.get("reviews", ""),
                    "store": product.get("store", ""),
                    "delivery": product.get("delivery", ""),
                    "condition": product.get("condition", ""),
                    "thumbnail": product.get("thumbnail", "")
                }
            })
        
        return {
            "answer": response["answer"],
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    if not model:
        return TranscriptionResponse(
            success=False,
            message="Whisper model not loaded correctly",
            text=None
        )
    
    try:
        # Create a temporary file to save the uploaded audio
        temp_dir = tempfile.gettempdir()
        file_id = str(uuid.uuid4())
        audio_path = os.path.join(temp_dir, f"recording_{file_id}.wav")
        
        # Save uploaded file to the temporary location
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Transcribe the audio file with whisper
        result = model.transcribe(audio_path)
        transcript = result["text"]
        
        # Clean up the temporary file
        try:
            os.remove(audio_path)
        except:
            pass
        
        return TranscriptionResponse(
            success=True,
            message="Transcription completed successfully",
            text=transcript
        )
    
    except Exception as e:
        return TranscriptionResponse(
            success=False,
            message=f"Error during transcription: {str(e)}",
            text=None
        )

@app.post("/voice-query", response_model=VoiceQueryResponse)
async def voice_query(
    file: UploadFile = File(...),
    user_id: str = Query(..., description="User ID for the query"),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    # Validate user_id
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    # First transcribe the audio
    transcription = await transcribe_audio(file)
    
    # If transcription failed, return just the transcription response
    if not transcription.success:
        return VoiceQueryResponse(
            transcription=transcription,
            query_response=None
        )
    
    # Get chat history for the user
    chat_history = await conversation_manager.get_chat_history(user_id)
    
    # Convert chat history to LangChain messages with proper validation
    lc_messages = []
    for msg in chat_history:
        # Check if msg is a dict (from get_chat_history) or a ChatHistoryItem
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
        
        # Only add valid messages
        if role and content and content.strip():
            if role.lower() in ["user", "you", "human"]:
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))
    
    # Get retrieval chain for this user
    retrieval_chain, intent = await get_retrieval_chain(user_id, transcription.text)
    
    # Process query
    try:
        # Save the transcribed query to history
        await conversation_manager.save_message(user_id, "user", transcription.text)
        
        # Directly invoke the retrieval chain
        response = retrieval_chain.invoke({
            "input": transcription.text,
            "chat_history": lc_messages
        })
        
        # Check if this is a recommendation query and check if exact item is present
        shopping_results = []
        if intent == "recommendation":
            # Extract search terms for checking exact matches
            search_terms, filters = extract_search_terms_and_filters_from_query(transcription.text)
            
            # Check if the exact item is present in database results
            exact_item_found = False
            if len(response["context"]) > 0:
                search_terms_lower = search_terms.lower()
                search_words = search_terms_lower.split()
                
                for doc in response["context"]:
                    doc_content = doc.page_content.lower()
                    # Check if any of the search terms appear in the document
                    if any(word in doc_content for word in search_words):
                        exact_item_found = True
                        break
            
            # Use Google Shopping only if exact item is not found in database
            if not exact_item_found:
                logger.info(f"Voice - Exact item '{search_terms}' not found in retailer database, searching Google Shopping")
                shopping_results = await search_google_shopping(search_terms, filters)
                
                if shopping_results:
                    # Add shopping results to the context for the LLM
                    shopping_context = format_shopping_results_for_llm(shopping_results)
                    
                    # Create a new prompt that includes shopping results
                    enhanced_system_prompt = """You are a smart expense assistant. Use these retailer receipts, conversation history, and shopping results to provide recommendations:

                    Retailer Database:
                    {context}

                    Shopping Results:
                    {shopping_context}

                    Conversation History:
                    {chat_history}

                    The exact item was not found in our retailer database, so here are current shopping options. Include prices, ratings, store names, and direct links from the shopping results. Be specific about where users can buy items online. Always mention the store name and include the product link in your response."""

                    enhanced_qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", enhanced_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])

                    enhanced_question_answer_chain = create_stuff_documents_chain(llm, enhanced_qa_prompt)
                    
                    # Re-invoke with shopping context
                    response = enhanced_question_answer_chain.invoke({
                        "input": transcription.text,
                        "context": response["context"],
                        "shopping_context": shopping_context,
                        "chat_history": lc_messages
                    })
                    
                    # Convert response format to match expected structure
                    if isinstance(response, str):
                        response = {
                            "answer": response,
                            "context": []
                        }
            else:
                logger.info(f"Voice - Exact item '{search_terms}' found in retailer database, using database results only")
        
        # Save the AI's response to history
        await conversation_manager.save_message(user_id, "assistant", response["answer"])
        
        # Format sources for the response
        sources = []
        for doc in response.get("context", []):
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        # Add shopping results as sources if they were used
        for product in shopping_results:
            sources.append({
                "content": f"Product: {product.get('title', '')} - Price: {product.get('price', '')}",
                "metadata": {
                    "source": "google_shopping",
                    "title": product.get("title", ""),
                    "price": product.get("price", ""),
                    "link": product.get("link", ""),
                    "rating": product.get("rating", ""),
                    "reviews": product.get("reviews", ""),
                    "store": product.get("store", ""),
                    "delivery": product.get("delivery", ""),
                    "condition": product.get("condition", ""),
                    "thumbnail": product.get("thumbnail", "")
                }
            })
        
        # Create the query response object
        query_response = QueryResponse(
            answer=response["answer"],
            sources=sources
        )
        
        return VoiceQueryResponse(
            transcription=transcription,
            query_response=query_response
        )
    except Exception as e:
        print(f"Error processing voice query: {str(e)}")
        # Return successful transcription but failed query
        return VoiceQueryResponse(
            transcription=transcription,
            query_response=None
        )
    

@app.get("/history/{user_id}")
async def get_chat_history(
    user_id: str,
    limit: int = Query(10, description="Maximum number of messages to return"),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Retrieve chat history for a user"""
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    messages = await conversation_manager.get_chat_history(user_id, max_messages=limit)
    return {"messages": messages}

@app.delete("/history/{user_id}")
async def clear_chat_history(
    user_id: str,
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Clear a user's conversation history"""
    deleted = await conversation_manager.clear_history(user_id)
    return {"success": True, "messages_deleted": deleted}

@app.post("/history/{user_id}/message")
async def add_message(
    user_id: str,
    role: str = Body(..., description="Message role (user or assistant)"),
    content: str = Body(..., description="Message content"),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Add a single message to the chat history"""
    if role not in ["user", "assistant"]:
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'assistant'")
    
    await conversation_manager.save_message(user_id, role, content)
    return {"success": True}

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Test connection to MongoDB
        client.admin.command('ping')
        whisper_loaded = model is not None
        searchapi_configured = bool(SEARCHAPI_KEY)
        return {
            "status": "healthy", 
            "database": "connected",
            "whisper_model": "loaded" if whisper_loaded else "not loaded",
            "searchapi": "configured" if searchapi_configured else "not configured"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("server-rec:app", host="0.0.0.0", port=8000, reload=True)