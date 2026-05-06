import streamlit as st
import requests
import io
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# --- Configuration ---
st.set_page_config(page_title="Audio ID System", page_icon="🎵", layout="wide")
API_URL = "http://localhost:8000"

# --- UI Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Audio ID System")
st.markdown("**Dual-Engine Audio Fingerprinting & Matching**")

def plot_audio_features(audio_data, sr, title="Audio Features"):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    
    # Waveform
    librosa.display.waveshow(audio_data, sr=sr, ax=ax[0])
    ax[0].set(title='Waveform')
    ax[0].label_outer()
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=ax[1])
    ax[1].set(title='Linear-frequency power spectrogram')
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    
    plt.tight_layout()
    return fig

def add_noise(audio_data, noise_level):
    # noise_level is between 0 and 1
    noise = np.random.randn(len(audio_data))
    augmented_data = audio_data + noise_level * noise
    # Normalize back to [-1, 1]
    if np.max(np.abs(augmented_data)) > 0:
        augmented_data = augmented_data / np.max(np.abs(augmented_data))
    return augmented_data

tab1, tab2 = st.tabs(["🔍 Identify Song", "📥 Ingest New Song"])

with tab1:
    st.header("Identify Audio")
    st.write("Upload a snippet or record from your mic to identify the song.")
    
    input_mode = st.radio("Input Method", ["Upload File", "Live Microphone"])
    
    query_file = None
    if input_mode == "Upload File":
        query_file = st.file_uploader("Upload Audio Snippet (WAV/MP3)", type=['wav', 'mp3'], key="query")
    else:
        query_file = st.audio_input("Record Audio Snippet")
        
    noise_level = st.slider("Distortion Simulator (Add Background Noise)", 0.0, 1.0, 0.0, 0.05,
                            help="Simulate a noisy environment by adding white noise to your recording.")
    
    if query_file is not None:
        st.subheader("Original Audio")
        st.audio(query_file)
        
        if st.button("Identify"):
            with st.spinner("Processing audio & extracting features..."):
                start_time = time.time()
                
                try:
                    # Load audio for processing/visualization
                    y, sr = librosa.load(io.BytesIO(query_file.getvalue()), sr=None)
                    
                    # Apply Distortion Simulator
                    if noise_level > 0:
                        y = add_noise(y, noise_level)
                        st.subheader("Distorted Audio (Sent to backend)")
                        
                        # Convert back to wav bytes to play in UI
                        buffer = io.BytesIO()
                        sf.write(buffer, y, sr, format='WAV')
                        buffer.seek(0)
                        st.audio(buffer, format='audio/wav')
                    
                    # Plot features
                    with st.expander("View Spectrogram & Waveform", expanded=True):
                        st.pyplot(plot_audio_features(y, sr))
                        
                    # Save modified (or original) audio to a buffer to send to API
                    out_buffer = io.BytesIO()
                    sf.write(out_buffer, y, sr, format='WAV')
                    out_buffer.seek(0)
                    
                    # Send to API
                    files = {"file": ("query.wav", out_buffer, "audio/wav")}
                    
                    st.text("Matching against database...")
                    response = requests.post(f"{API_URL}/identify/", files=files)
                    result = response.json()
                    
                    elapsed = time.time() - start_time
                    
                    if result.get("match"):
                        st.success(f"Match Found! ({elapsed:.2f}s)")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Title", result.get("title"))
                        col2.metric("Artist", result.get("artist"))
                        
                        conf = result.get("confidence", 0) * 100
                        col3.metric("Confidence", f"{conf:.1f}%")
                        
                        st.info(f"Engine Used: **{result.get('engine').upper()}**")
                        
                        st.progress(int(conf))
                    else:
                        st.warning("No match found in the database.")
                except Exception as e:
                    st.error(f"Error processing or connecting to backend: {e}")

with tab2:
    st.header("Ingest Song into Database")
    st.write("Upload a full song to index it into the fingerprint database.")
    
    ingest_file = st.file_uploader("Upload Full Audio (WAV/MP3)", type=['wav', 'mp3'], key="ingest")
    title = st.text_input("Song Title")
    artist = st.text_input("Artist")
    
    if ingest_file and title and artist:
        if st.button("Ingest Song"):
            with st.spinner("Extracting fingerprints and embeddings..."):
                files = {"file": (ingest_file.name, ingest_file.getvalue(), ingest_file.type)}
                data = {"title": title, "artist": artist}
                
                try:
                    response = requests.post(f"{API_URL}/ingest/", files=files, data=data)
                    result = response.json()
                    
                    if response.status_code == 200:
                        st.success(f"Successfully ingested '{title}'!")
                        st.write(f"Generated **{result.get('hashes_extracted')}** hashes.")
                    else:
                        st.error(f"Error: {result.get('detail')}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

# Formatted
