import streamlit as st
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import time
import queue
import threading
import whisper

# Set page config
st.set_page_config(page_title="Voice Transcription App", page_icon="🎤")

# Define constants
SAMPLE_RATE = 16000
CHANNELS = 1

# Initialize session state variables if they don't exist
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_filename' not in st.session_state:
    st.session_state.audio_filename = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

# Title and description
st.title("Voice Recording and Transcription")
st.write("Record your voice and get it transcribed using OpenAI's Whisper")

# Initialize status placeholder
status_placeholder = st.empty()

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

model = load_whisper_model()

# Audio recording functions
def start_recording_click():
    """Called when the 'Start Recording' button is clicked"""
    # Reset state
    st.session_state.recording = True
    st.session_state.transcript = ""
    st.session_state.audio_filename = None
    st.session_state.audio_bytes = None
    status_placeholder.info("Recording... Click 'Stop Recording' when finished.")
    st.rerun()

def stop_recording_click():
    """Called when the 'Stop Recording' button is clicked"""
    st.session_state.recording = False
    status_placeholder.info("Processing audio...")
    st.rerun()

# Record audio in non-threaded way (safer with Streamlit)
if st.session_state.recording and not st.session_state.audio_filename:
    # Create a file to save the recording
    temp_dir = tempfile.gettempdir()
    audio_filename = os.path.join(temp_dir, f"recording_{int(time.time())}.wav")
    
    # Create a queue to store audio chunks
    audio_queue = queue.Queue()
    
    # Record settings
    duration = 30  # Maximum recording duration in seconds
    
    # Status for showing progress
    recording_status = status_placeholder.info(f"Recording... (Max {duration}s)")
    
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
            
            while st.session_state.recording and (time.time() - start_time) < duration:
                # Try to get chunk from queue
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    audio_chunks.append(chunk)
                except queue.Empty:
                    pass
                
                # Update progress bar
                elapsed = time.time() - start_time
                progress_bar.progress(min(elapsed / duration, 1.0))
                
                # Update status message with elapsed time
                recording_status.info(f"Recording... {elapsed:.1f}s / {duration}s (Click 'Stop Recording' when finished)")
                
                # Very brief sleep to allow for UI updates
                time.sleep(0.01)
                
                # Check if recording should be stopped
                if not st.session_state.recording or elapsed >= duration:
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
            if (time.time() - start_time) >= duration:
                st.session_state.recording = False
            
            # Transcribe the audio
            status_placeholder.info("Transcribing audio...")
            
            # Transcribe the audio file with whisper
            try:
                if model:
                    result = model.transcribe(audio_filename)
                    st.session_state.transcript = result["text"]
                    status_placeholder.success("Transcription complete!")
                else:
                    status_placeholder.error("Whisper model not loaded correctly. Cannot transcribe.")
            except Exception as e:
                status_placeholder.error(f"Error during transcription: {e}")
        else:
            status_placeholder.warning("No audio recorded.")
            st.session_state.recording = False
        
    except Exception as e:
        st.error(f"Error during recording: {e}")
        st.session_state.recording = False

# Create two columns for the start and stop buttons
col1, col2 = st.columns(2)

# Start recording button
with col1:
    st.button("Start Recording", 
              type="primary", 
              on_click=start_recording_click,
              disabled=st.session_state.recording)

# Stop recording button
with col2:
    st.button("Stop Recording", 
              type="secondary",
              on_click=stop_recording_click,
              disabled=not st.session_state.recording)

# Display the transcript
if st.session_state.transcript:
    st.subheader("Transcription:")
    st.write(st.session_state.transcript)
    
    # Play back the recorded audio
    if st.session_state.audio_bytes:
        st.audio(st.session_state.audio_bytes)

# Add some information at the bottom
st.markdown("---")
st.markdown("This app uses OpenAI's Whisper for transcription.")
st.markdown("Make sure your microphone is properly connected and permissions are granted.")