import streamlit as st
import requests
import io
import time

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

tab1, tab2 = st.tabs(["🔍 Identify Song", "📥 Ingest New Song"])

with tab1:
    st.header("Identify Audio")
    st.write("Upload a snippet of audio (even with background noise) to identify it.")
    
    query_file = st.file_uploader("Upload Audio Snippet (WAV/MP3)", type=['wav', 'mp3'], key="query")
    
    if query_file is not None:
        st.audio(query_file)
        
        if st.button("Identify"):
            with st.spinner("Analyzing audio..."):
                start_time = time.time()
                
                # Send to API
                files = {"file": (query_file.name, query_file.getvalue(), query_file.type)}
                try:
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
                    st.error(f"Error connecting to backend: {e}")

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

# formatted
