import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Body, Query
from pydantic import BaseModel, Field
import uvicorn

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
    title="AI Expense Assistant API",
    description="API for querying expense data using natural language",
    version="1.0.0",
)

# Initialize embeddings and MongoDB connection
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    task_type="RETRIEVAL_QUERY"
)
client = MongoClient(MONGO_URI)
db = client["bem"]
collection = db["flattened_expenses_googleai"]
conversations_collection = db["user_conversations"]

# Create index for faster conversation querying
conversations_collection.create_index([("user_id", 1), ("timestamp", -1)])

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="receipts_vector_index"
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
    return {"message": "Welcome to the AI Expense Assistant API"}

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
    
    # Convert chat history to LangChain messages
    lc_messages = []
    for msg in chat_history:
        if msg.role.lower() in ["user", "you", "human"]:
            lc_messages.append(HumanMessage(content=msg.content))
        else:
            lc_messages.append(AIMessage(content=msg.content))
    
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
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)