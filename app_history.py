# Importing the relevant libraries
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # GoogleGenerativeAIEmbeddings: turns text into vectors (for searching).ChatGoogleGenerativeAI: allows chatting with Gemini.
from langchain_mongodb import MongoDBAtlasVectorSearch # MongoDBAtlasVectorSearch: Lets you search similar texts (using vector embeddings) in MongoDB.
from pymongo import MongoClient # MongoClient: Connects to your MongoDB database. 
from langchain.chains import create_retrieval_chain, create_history_aware_retriever # Retrieval chains get relevant documents. History-aware retrievers consider past chat history.
from langchain.chains.combine_documents import create_stuff_documents_chain # "Stuff" chains combine receipts into answers.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # MessagesPlaceholder allows inserting full conversation history.
from langchain_core.messages import HumanMessage, AIMessage # HumanMessage and AIMessage help differentiate who said what.
from streamlit_mic_recorder import speech_to_text  # For mic recording and speech-to-text

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize embeddings and MongoDB connection

# Converts your query into embeddings (numerical format for vector search).
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    task_type="RETRIEVAL_QUERY"
)
client = MongoClient(MONGO_URI)
collection = client["bem"]["flattened_expenses_googleai"]

# Sets up vector search on the receipt data using embeddings.
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="receipts_vector_index"
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3 # controls degree of randomness 
)

# Contextual retriever chain
retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Creates a retriever that fetches top 5 similar receipts to your question.

# This prompt helps the model rewrite the user's current question based on chat history.
retriever_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("human", "Given the conversation history, reformulate this as a standalone query about expenses:"),
])

# Combines the above prompt + retriever to handle chat context smartly.
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, retriever_prompt
)

# Answer generation chain
system_prompt = """You are a smart expense assistant. Use these receipts and conversation history:

Receipts:
{context}

Conversation History:
{chat_history}"""

# Formats the final input to the model with receipts, history, and user query.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Combines retriever and answer-generator into a final pipeline (chain).
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Streamlit UI
st.set_page_config(page_title="AI Expense Assistant", page_icon="🧾")
st.title("AI Expense Assistant 🧾")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Audio Input Section ---
st.subheader("🎤 Speak your question")
audio_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='stt')
if audio_text:
    st.success(f"Transcribed: {audio_text}")
    query_text = audio_text
else:
    query_text = None

# --- Text Input Section ---
st.subheader("💬 Or type your question")
typed_query = st.chat_input("Ask about expenses...")

if typed_query:
    query_text = typed_query
    
if query_text:
    # Convert chat history to LangChain messages
    lc_messages = [
        HumanMessage(content=content) if role == "You" 
        else AIMessage(content=content) 
        for role, content in st.session_state.chat_history
    ]
    
    with st.spinner("Analyzing..."):
        response = retrieval_chain.invoke({
            "input": query_text,
            "chat_history": lc_messages
        })
        
        # Update chat history
        st.session_state.chat_history.extend([
            ("You", query_text),
            ("AI", response["answer"])
        ])

    # Display sources
    with st.expander("Source Receipts"):
        for doc in response["context"]:
            st.write(doc.page_content)

# Display conversation
for role, content in st.session_state.chat_history[-6:]:
    st.chat_message(role).write(content)
