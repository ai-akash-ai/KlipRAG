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

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient, DESCENDING
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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
    transcription_time: Optional[float] = None

class VoiceQueryResponse(BaseModel):
    transcription: TranscriptionResponse
    query_response: Optional[QueryResponse] = None
    llm_processing_time: Optional[float] = None

class MessageRequest(BaseModel):
    role: str
    content: str

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
    
# Create retrieval chain factory
def get_retrieval_chain(user_id: str):
    # Contextual retriever with user_id filter
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5,
            "pre_filter": {
                "metadata.user_id": user_id
            }
        }
    )

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("human", "Given the conversation history, reformulate this as a standalone query about expenses:"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, retriever_prompt
    )

    # Answer generation chain
    system_prompt = """You are a smart expense assistant. Use these receipts and conversation history:

    Receipts:
    {context}

    Conversation History:
    {chat_history}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

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
    retrieval_chain = get_retrieval_chain(request.user_id)
    
    # Process query
    try:
        # Save the user's query to history
        await conversation_manager.save_message(request.user_id, "user", request.query)
        
        response = retrieval_chain.invoke({
            "input": request.query,
            "chat_history": lc_messages
        })
        
        # Save the AI's response to history
        await conversation_manager.save_message(request.user_id, "assistant", response["answer"])
        
        # Format sources for the response
        sources = []
        for doc in response["context"]:
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata
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
            text=None,
            transcription_time=None
        )
    
    try:
        # Start timing for transcription
        transcription_start_time = time.time()
        
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
        
        # Calculate transcription time
        transcription_time = time.time() - transcription_start_time
        
        # Clean up the temporary file
        try:
            os.remove(audio_path)
        except:
            pass
        
        return TranscriptionResponse(
            success=True,
            message="Transcription completed successfully",
            text=transcript,
            transcription_time=round(transcription_time, 3)
        )
    
    except Exception as e:
        transcription_time = time.time() - transcription_start_time if 'transcription_start_time' in locals() else None
        return TranscriptionResponse(
            success=False,
            message=f"Error during transcription: {str(e)}",
            text=None,
            transcription_time=round(transcription_time, 3) if transcription_time else None
        )

@app.post("/voice-query", response_model=VoiceQueryResponse)
async def voice_query(
    user_id: str = Query(..., description="User ID for the query"),
    file: UploadFile = File(...),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    # Validate user_id
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
        
    # STEP 1: Transcription Process
    print("Starting transcription process...")
    transcription = await transcribe_audio(file)
    
    # If transcription failed, return just the transcription response
    if not transcription.success:
        return VoiceQueryResponse(
            transcription=transcription,
            query_response=None,
            llm_processing_time=None
        )
    
    print(f"Transcription completed in {transcription.transcription_time}s: {transcription.text}")
    
    # STEP 2: LLM Processing
    print("Starting LLM processing...")
    llm_start_time = time.time()
    
    try:
        # Get chat history for the user
        chat_history = await conversation_manager.get_chat_history(user_id)
        
        # Convert chat history to LangChain messages
        lc_messages = []
        for msg in chat_history:
            # Check if msg is a dict (from get_chat_history) or a ChatHistoryItem
            if isinstance(msg, dict):
                role = msg["role"]
                content = msg["content"]
            else:
                role = msg.role
                content = msg.content
                
            if role.lower() in ["user", "you", "human"]:
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))
        
        # Get retrieval chain for this user
        retrieval_chain = get_retrieval_chain(user_id)
        
        # Save the transcribed query to history
        await conversation_manager.save_message(user_id, "user", transcription.text)
        
        # Directly invoke the retrieval chain (same as in query_expenses)
        response = retrieval_chain.invoke({
            "input": transcription.text,
            "chat_history": lc_messages
        })
        
        # Calculate LLM processing time
        llm_processing_time = time.time() - llm_start_time
        
        # Save the AI's response to history
        await conversation_manager.save_message(user_id, "assistant", response["answer"])
        
        # Format sources for the response
        sources = []
        for doc in response["context"]:
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Create the query response object
        query_response = QueryResponse(
            answer=response["answer"],
            sources=sources
        )
        
        print(f"LLM processing completed in {llm_processing_time:.3f}s")
        
        return VoiceQueryResponse(
            transcription=transcription,
            query_response=query_response,
            llm_processing_time=round(llm_processing_time, 3)
        )
        
    except Exception as e:
        llm_processing_time = time.time() - llm_start_time
        print(f"Error processing voice query: {str(e)}")
        # Return successful transcription but failed query
        return VoiceQueryResponse(
            transcription=transcription,
            query_response=None,
            llm_processing_time=round(llm_processing_time, 3)
        )

@app.get("/history", response_model=Dict[str, List[Dict[str, str]]])
async def get_chat_history(
    user_id: str = Query(..., description="User ID to retrieve history for"),
    limit: int = Query(10, description="Maximum number of messages to return"),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Retrieve chat history for a user"""
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    messages = await conversation_manager.get_chat_history(user_id, max_messages=limit)
    return {"messages": messages}

@app.delete("/history", response_model=Dict[str, Any])
async def clear_chat_history(
    user_id: str = Query(..., description="User ID to clear history for"),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Clear a user's conversation history"""
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
        
    deleted = await conversation_manager.clear_history(user_id)
    return {"success": True, "messages_deleted": deleted}

@app.post("/history/message", response_model=Dict[str, bool])
async def add_message(
    user_id: str = Query(..., description="User ID to add message to"),
    message: MessageRequest = Body(...),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Add a single message to the chat history"""
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
        
    if message.role not in ["user", "assistant"]:
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'assistant'")
    
    await conversation_manager.save_message(user_id, message.role, message.content)
    return {"success": True}

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Test connection to MongoDB
        client.admin.command('ping')
        whisper_loaded = model is not None
        return {
            "status": "healthy", 
            "database": "connected",
            "whisper_model": "loaded" if whisper_loaded else "not loaded"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("final-server:app", host="0.0.0.0", port=8000, reload=True)