import streamlit as st
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import time
import queue
import requests
import io
import json
from datetime import datetime
import uuid

# Set page config
st.set_page_config(
    page_title="AI Expense Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://localhost:8000"  # FastAPI backend URL

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RECORDING_DURATION = 10  # seconds

# Initialize session state variables
if 'user_id' not in st.session_state:
    # Generate a user ID if not already stored
    st.session_state.user_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_filename' not in st.session_state:
    st.session_state.audio_filename = None
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'sources' not in st.session_state:
    st.session_state.sources = []

# Sidebar
with st.sidebar:
    st.title("💰 Expense Assistant")
    st.write("Ask questions about your expenses using text or voice.")
    
    # User ID display and option to change
    st.subheader("User Settings")
    user_id_display = st.text_input("Your User ID", value=st.session_state.user_id)
    if user_id_display != st.session_state.user_id:
        st.session_state.user_id = user_id_display
        st.session_state.messages = []  # Clear messages when changing user
        st.rerun()
    
    # Clear chat history
    if st.button("Clear Chat History"):
        try:
            response = requests.delete(f"{API_URL}/history?user_id={st.session_state.user_id}")
            if response.status_code == 200:
                st.session_state.messages = []
                st.success("Chat history cleared!")
            else:
                st.error("Failed to clear chat history.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Check API health
    st.subheader("API Status")
    if st.button("Check API Status"):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                health_data = response.json()
                st.success(f"API Status: {health_data['status']}")
                st.info(f"Database: {health_data['database']}")
                st.info(f"Whisper Model: {health_data['whisper_model']}")
            else:
                st.error("API is available but returned an error")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure the FastAPI backend is running.")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This application allows you to query your expenses using natural language. You can type your questions or use voice input.")
    st.markdown("Powered by LangChain, Google Gemini, and OpenAI Whisper.")

# Main content
st.title("AI Expense Assistant")

# Initialize status placeholder
status_placeholder = st.empty()

# Function to load chat history from API
def load_chat_history():
    try:
        response = requests.get(f"{API_URL}/history?user_id={st.session_state.user_id}")
        if response.status_code == 200:
            history_data = response.json()
            st.session_state.messages = history_data["messages"]
    except Exception as e:
        status_placeholder.error(f"Error loading chat history: {str(e)}")

# Check if we need to load messages from API
if len(st.session_state.messages) == 0:
    load_chat_history()

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

# Function to add a message to the chat
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# Function to process text query
def process_query(query):
    status_placeholder.info("Processing your query...")
    
    try:
        # Prepare the request payload
        payload = {
            "user_id": st.session_state.user_id,
            "query": query,
            "chat_history": st.session_state.messages
        }
        
        # Send the request to the API
        response = requests.post(
            f"{API_URL}/query",
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Store the sources for display
            st.session_state.sources = result["sources"]
            
            # Add the messages to the chat
            add_message("user", query)
            add_message("assistant", result["answer"])
            
            # Display the assistant's response
            with st.chat_message("assistant"):
                st.write(result["answer"])
            
            status_placeholder.empty()
            return True
        else:
            status_placeholder.error(f"API Error: {response.status_code}")
            return False
    except Exception as e:
        status_placeholder.error(f"Error processing query: {str(e)}")
        return False

# Voice recording functions
def start_recording_click():
    """Called when the 'Start Recording' button is clicked"""
    # Reset state
    st.session_state.recording = True
    st.session_state.audio_filename = None
    st.session_state.audio_bytes = None
    status_placeholder.info("Recording... Click 'Stop Recording' when finished.")
    st.rerun()

def stop_recording_click():
    """Called when the 'Stop Recording' button is clicked"""
    st.session_state.recording = False
    status_placeholder.info("Processing audio...")
    st.rerun()

# Handle voice recording
def handle_voice_recording():
    if st.session_state.recording and not st.session_state.audio_filename:
        # Create a file to save the recording
        temp_dir = tempfile.gettempdir()
        audio_filename = os.path.join(temp_dir, f"recording_{int(time.time())}.wav")
        
        # Create a queue to store audio chunks
        audio_queue = queue.Queue()
        
        # Status for showing progress
        recording_status = status_placeholder.info(f"Recording... (Max {MAX_RECORDING_DURATION}s)")
        
        # Define callback function for the audio stream
        def callback(indata, frames, time, status):
            if status:
                print(status)
            audio_queue.put(indata.copy())
        
        # Use a try-except block to handle potential sounddevice errors
        try:
            # Start recording with sounddevice
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Record until stop button is pressed or max duration is reached
                audio_chunks = []
                start_time = time.time()
                
                while st.session_state.recording and (time.time() - start_time) < MAX_RECORDING_DURATION:
                    # Try to get chunk from queue
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                        audio_chunks.append(chunk)
                    except queue.Empty:
                        pass
                    
                    # Update progress bar
                    elapsed = time.time() - start_time
                    progress_bar.progress(min(elapsed / MAX_RECORDING_DURATION, 1.0))
                    
                    # Update status message with elapsed time
                    recording_status.info(f"Recording... {elapsed:.1f}s / {MAX_RECORDING_DURATION}s (Click 'Stop Recording' when finished)")
                    
                    # Very brief sleep to allow for UI updates
                    time.sleep(0.01)
                    
                    # Check if recording should be stopped
                    if not st.session_state.recording or elapsed >= MAX_RECORDING_DURATION:
                        break
            
            # Save the audio
            if audio_chunks:
                audio_data = np.concatenate(audio_chunks, axis=0)
                sf.write(audio_filename, audio_data, SAMPLE_RATE)
                
                # Read the file into memory to display later
                with open(audio_filename, 'rb') as f:
                    audio_bytes = f.read()
                
                # Update session state with file info
                st.session_state.audio_filename = audio_filename
                st.session_state.audio_bytes = audio_bytes
                
                # If we reached the end naturally (not by pressing Stop), update the recording state
                if (time.time() - start_time) >= MAX_RECORDING_DURATION:
                    st.session_state.recording = False
                
                # Send the audio to the API for voice query
                status_placeholder.info("Sending audio to API for transcription and processing...")
                
                try:
                    # Prepare the file for upload
                    files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
                    
                    # Send the audio to the voice-query endpoint with user_id as a query parameter
                    response = requests.post(
                        f"{API_URL}/voice-query?user_id={st.session_state.user_id}",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        transcription = result.get("transcription", {})
                        query_response = result.get("query_response", None)
                        
                        if transcription.get("success", False):
                            transcribed_text = transcription.get("text", "")
                            
                            # Display the transcribed text as user message
                            add_message("user", transcribed_text)
                            with st.chat_message("user"):
                                st.write(transcribed_text)
                                
                            # Display a player for the recorded audio
                            st.audio(audio_bytes, format="audio/wav")
                            
                            if query_response:
                                # Store the sources
                                st.session_state.sources = query_response.get("sources", [])
                                
                                # Display the assistant's response
                                answer = query_response.get("answer", "")
                                add_message("assistant", answer)
                                with st.chat_message("assistant"):
                                    st.write(answer)
                                
                                status_placeholder.success("Voice query processed successfully!")
                            else:
                                status_placeholder.warning("Transcription succeeded but query processing failed.")
                        else:
                            status_placeholder.error(f"Transcription failed: {transcription.get('message')}")
                    else:
                        status_placeholder.error(f"API returned status code {response.status_code}")
                except Exception as e:
                    status_placeholder.error(f"Error processing voice query: {e}")
            else:
                status_placeholder.warning("No audio recorded.")
                st.session_state.recording = False
            
        except Exception as e:
            st.error(f"Error during recording: {e}")
            st.session_state.recording = False

# Check for voice recording status and handle it
handle_voice_recording()

# Display source documents in an expander if available
if st.session_state.sources:
    with st.expander("Source Documents"):
        for i, source in enumerate(st.session_state.sources):
            st.markdown(f"**Source {i+1}**")
            
            # Display metadata if available
            if "metadata" in source:
                metadata = source["metadata"]
                if "date" in metadata:
                    st.markdown(f"**Date:** {metadata.get('date')}")
                if "merchant" in metadata:
                    st.markdown(f"**Merchant:** {metadata.get('merchant')}")
                if "total_amount" in metadata:
                    st.markdown(f"**Amount:** ${metadata.get('total_amount')}")
                if "category" in metadata:
                    st.markdown(f"**Category:** {metadata.get('category')}")
            
            # Display content
            st.markdown("**Content:**")
            st.markdown(source.get("content", ""))
            st.markdown("---")

# Input options area
st.markdown("## Ask a Question")
st.write("You can type your question or use voice input.")

# Create tabs for text and voice input
tab1, tab2 = st.tabs(["Text Query", "Voice Query"])

with tab1:
    # Text input
    query = st.text_input("Type your question about expenses:")
    if st.button("Submit", key="text_submit"):
        if query:
            process_query(query)
        else:
            st.warning("Please enter a question.")

with tab2:
    # Voice input buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.button(
            "Start Recording", 
            type="primary", 
            on_click=start_recording_click,
            disabled=st.session_state.recording,
            key="start_record"
        )
    
    with col2:
        st.button(
            "Stop Recording", 
            type="secondary",
            on_click=stop_recording_click,
            disabled=not st.session_state.recording,
            key="stop_record"
        )
    
    st.write("Click 'Start Recording' and ask your question. Click 'Stop Recording' when finished.")

# Chat input at the bottom (alternative to the tabs)
if query := st.chat_input("Type a question here..."):
    process_query(query)

# Footer
st.markdown("---")
st.markdown("AI Expense Assistant | Powered by LangChain, Google Gemini, and OpenAI Whisper")