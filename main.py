import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from pytube import YouTube
# from youtube_transcript_api import (
#     YouTubeTranscriptApi,
#     TranscriptsDisabled, 
#     NoTranscriptFound,
#     VideoUnavailable,                                                                                                           
#     CouldNotRetrieveTranscript

# )

import yt_dlp
import json
import re

from dotenv import load_dotenv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Smart RAG YouTube Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: #f0f4f8;
        color: #333;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #ffc107;
        font-weight: bold;
    }
    
    .answer-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'groq_usage_count' not in st.session_state:
    st.session_state.groq_usage_count = 0
if 'rag_confidence_threshold' not in st.session_state:
    st.session_state.rag_confidence_threshold = 0.3

@st.cache_resource
def load_embeddings():
    """Load HuggingFace embeddings for RAG"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# def get_youtube_transcript(url):
#     try:
#         video_id = YouTube(url).video_id
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
#         transcript = transcript_list.find_transcript(["en"])
#         transcript_data = transcript.fetch()
        
#         text_parts = []
#         for item in transcript_data:
#             if hasattr(item, 'text'):
#                 text_parts.append(item.text)
#             elif isinstance(item, dict) and 'text' in item:
#                 text_parts.append(item['text'])
#             else:
#                 text_parts.append(str(item))
        
#         return " ".join(text_parts)
#     except Exception as e:
#         st.error(f"Error getting transcript: {e}")
#         return ""

# # def get_youtube_transcript(url):
# #     try:
# #         video_id = YouTube(url).video_id
# #         # Directly get the transcript in English
# #         transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

# #         text_parts = [item['text'] for item in transcript_data if 'text' in item]
# #         return " ".join(text_parts)

# #     except TranscriptsDisabled:
# #         st.error("Transcripts are disabled for this video.")
# #     except NoTranscriptFound:
# #         st.error("No transcript found for this video.")
# #     except VideoUnavailable:
# #         transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

# #         text_parts = [item['text'] for item in transcript_data if 'text' in item]
# #         return " ".join(text_parts)

# #     except TranscriptsDisabled:
# #         st.error("Transcripts are disabled for this video.")
# #     except NoTranscriptFound:
# #         st.error("No transcript found for this video.")
# #     except VideoUnavailable:
# #         st.error("The video is unavailable.")
# #     except CouldNotRetrieveTranscript:
# #         st.error("Could not retrieve the transcript.")
# #     except Exception as e:
# #         st.error(f"Error getting transcript: {e}")
# #     return ""

# def get_youtube_transcript(url):
#     try:
#         video_id = YouTube(url).video_id
#         # Directly get the transcript in English
#         transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

#         text_parts = [item['text'] for item in transcript_data if 'text' in item]
#         return " ".join(text_parts)

#     except TranscriptsDisabled:
#         st.error("Transcripts are disabled for this video.")
#     except NoTranscriptFound:
#         st.error("No transcript found for this video.")
#     except VideoUnavailable:
#         st.error("The video is unavailable.")
#     except CouldNotRetrieveTranscript:
#         st.error("Could not retrieve the transcript.")
#     except Exception as e:
#         st.error(f"Error getting transcript: {e}")
#     return ""

def parse_youtube_json_transcript(raw_text):
    """Parse YouTube's JSON transcript format"""
    try:
        # Try to parse as JSON
        data = json.loads(raw_text)
        
        text_parts = []
        
        # Navigate the JSON structure
        if 'events' in data:
            for event in data['events']:
                if 'segs' in event:
                    for seg in event['segs']:
                        if 'utf8' in seg:
                            text = seg['utf8'].strip()
                            # Skip newlines and empty segments
                            if text and text != '\n':
                                text_parts.append(text)
        
        # Join and clean the text
        clean_text = ' '.join(text_parts)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
        
    except:
        # If JSON parsing fails, try VTT parsing
        return parse_vtt_format(raw_text)

def parse_vtt_format(vtt_text):
    """Parse VTT subtitle format"""
    try:
        lines = vtt_text.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip timestamps, headers, and empty lines
            if (not line or 
                '-->' in line or 
                line.startswith('WEBVTT') or
                line.startswith('Kind:') or
                line.startswith('Language:') or
                re.match(r'^\d{2}:\d{2}:\d{2}', line) or
                line.isdigit()):
                continue
            
            # Remove HTML tags
            clean_line = re.sub(r'<[^>]+>', '', line)
            if clean_line:
                text_lines.append(clean_line)
        
        full_text = ' '.join(text_lines)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        return full_text
            
    except Exception as e:
        return vtt_text  # Return original if parsing fails

def get_youtube_transcript(url):
    """Get YouTube transcript using yt-dlp with proper JSON parsing"""
    try:
        st.info("🔄 Extracting transcript using yt-dlp...")
        
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video info
            info = ydl.extract_info(url, download=False)
            
            st.success(f"🎥 Video: {info.get('title', 'Unknown')}")
            
            # Get subtitles and auto-captions
            subtitles = info.get('subtitles', {})
            auto_captions = info.get('automatic_captions', {})
            
            transcript_text = None
            
            # Try manual subtitles first
            if 'en' in subtitles:
                subtitle_url = subtitles['en'][0]['url']
                response = ydl.urlopen(subtitle_url)
                transcript_text = response.read().decode('utf-8')
                st.success("✅ Found manual English subtitles")
                
            # Try auto-generated captions
            elif 'en' in auto_captions:
                subtitle_url = auto_captions['en'][0]['url']
                response = ydl.urlopen(subtitle_url)
                transcript_text = response.read().decode('utf-8')
                st.success("✅ Found auto-generated English captions")
            
            else:
                available_subs = list(subtitles.keys()) + list(auto_captions.keys())
                if available_subs:
                    st.warning(f"⚠️ No English subtitles. Available: {', '.join(set(available_subs))}")
                else:
                    st.error("❌ No subtitles available for this video")
                return ""
            
            if transcript_text:
                # Parse the transcript based on format
                if transcript_text.strip().startswith('{'):
                    # JSON format
                    clean_text = parse_youtube_json_transcript(transcript_text)
                else:
                    # VTT format
                    clean_text = parse_vtt_format(transcript_text)
                
                if clean_text and len(clean_text) > 50:
                    st.success(f"✅ Extracted {len(clean_text)} characters of clean transcript")
                    return clean_text
                else:
                    st.error("❌ Could not extract readable text from transcript")
                    return ""
            
    except Exception as e:
        st.error(f"❌ Error extracting transcript: {str(e)}")
        return ""
def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def setup_rag_system(transcript_text):
    """Set up RAG system with transcript"""
    try:
        # Save transcript
        save_transcript_to_file(transcript_text)
        
        # Load and split documents
        loader = TextLoader("transcript.txt", encoding="utf-8")
        documents = loader.load()
        
        # Use smaller chunks for better precision
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = load_embeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        return vectorstore, docs
    except Exception as e:
        st.error(f"Error setting up RAG: {e}")
        return None, None

def calculate_rag_confidence(question, retrieved_docs, transcript_text):
    """Calculate how confident we are in the RAG answer"""
    try:
        # Combine retrieved documents
        combined_docs = " ".join([doc.page_content for doc in retrieved_docs])
        
        # Use TF-IDF to calculate similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([question, combined_docs])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Check if question words are well represented in retrieved docs
        question_words = set(question.lower().split())
        doc_words = set(combined_docs.lower().split())
        word_overlap = len(question_words.intersection(doc_words)) / len(question_words)
        
        # Combined confidence score
        confidence = (similarity * 0.6) + (word_overlap * 0.4)
        
        return confidence, combined_docs
    except Exception as e:
        return 0.0, ""

def get_rag_answer(question, vectorstore):
    """Get answer using RAG (local search only)"""
    try:
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(question)
        
        # Calculate confidence
        confidence, combined_context = calculate_rag_confidence(question, relevant_docs, "")
        
        # Generate simple answer from context
        if confidence > st.session_state.rag_confidence_threshold:
            # Extract most relevant sentences
            sentences = combined_context.split('.')
            relevant_sentences = []
            
            question_words = question.lower().split()
            for sentence in sentences:
                if any(word in sentence.lower() for word in question_words if len(word) > 2):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                answer = ". ".join(relevant_sentences[:3])
                return answer, confidence, combined_context
        
        return None, confidence, combined_context
    except Exception as e:
        return None, 0.0, ""

# API Key setup
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("🔑 Please set your GROQ_API_KEY in the .env file!")
    st.stop()

def get_groq_answer(question, context, groq_api_key):
    """Get answer from Groq with context"""
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-70b-8192",
            temperature=0.1
        )
        
        prompt = f"""Based on the following video transcript context, answer the question accurately and concisely.

Context from video transcript:
{context[:2000]}...

Question: {question}

Please provide a clear, informative answer based on the context. If the question cannot be answered from the context, say so clearly."""

        response = llm.invoke(prompt)
        st.session_state.groq_usage_count += 1
        return response.content
    except Exception as e:
        st.error(f"Groq Error Details: {str(e)}")
        return f"Error with Groq: {str(e)}"

def get_enhanced_answer(question, context, groq_api_key):
    """Get enhanced answer combining RAG context with Groq"""
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-70b-8192",
            temperature=0.1
        )
        
        prompt = f"""You are an expert tutor. Based on the video transcript context below, provide a comprehensive answer to the question.

Video transcript context:
{context[:2000]}...

Question: {question}

Please provide:
1. A clear, detailed answer based on the video content
2. Key insights or explanations that help understand the topic
3. Any relevant examples mentioned in the video

Answer:"""

        response = llm.invoke(prompt)
        st.session_state.groq_usage_count += 1
        
        return response.content
    except Exception as e:
        return f"Error with Groq: {e}"

# Main UI Layout
st.markdown("""
<div class="main-header">
    <h1> Smart RAG +  YouTube Tutor</h1>
    <p>Efficient AI tutoring that saves API calls by using local search first!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Settings
    st.markdown("### 🎛️ RAG Settings")
    st.session_state.rag_confidence_threshold = st.slider(
        "Confidence Threshold", 
        0.0, 1.0, 0.3, 0.1,
        help="Higher = More likely to use Groq. Lower = More RAG-only answers"
    )
    
    st.markdown("### 🤖 Answer Mode")
    answer_mode = st.radio(
        "Select Mode:",
        ["🔍 Smart Mode", "🚀 Enhanced Mode", "💾 RAG Only"],
        help="Choose how the system should generate answers"
    )
    
    # # Usage Statistics
    # st.markdown("### 📊 Usage Statistics")
    # st.markdown(f"""
    # <div class="metric-card">
    #     <h3>{st.session_state.groq_usage_count}</h3>
    #     <p>Groq API Calls Used</p>
    # </div>
    # """, unsafe_allow_html=True)
    
    # # How it works
    # st.markdown("### 🎯 How It Works")
    # st.markdown("""
    # <div class="feature-card">
    #     <strong>1. RAG First</strong><br>
    #     Search transcript locally for instant results
    # </div>
    # <div class="feature-card">
    #     <strong>2. Smart Decision</strong><br>
    #     Use Groq only when needed based on confidence
    # </div>
    # <div class="feature-card">
    #     <strong>3. Save API Calls</strong><br>
    #     Efficient resource usage and cost optimization
    # </div>
    # """, unsafe_allow_html=True)

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎥 Video Input")
    video_url = st.text_input(
        "YouTube Video URL:", 
        placeholder="https://www.youtube.com/watch?v=...",
        key="video_url_input"
    )

with col2:
    st.markdown("### Click to Process Video  ")
    st.markdown("")
    process_button = st.button("🚀 Process Video", type="primary", use_container_width=True)

# Video Processing
if process_button:
    if not video_url:
        st.markdown('<div class="status-error">❌ Please enter a YouTube URL!</div>', unsafe_allow_html=True)
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Get transcript
        status_text.text("📝 Getting transcript...")
        progress_bar.progress(25)
        transcript_text = get_youtube_transcript(video_url)
        
        if transcript_text:
            st.session_state.transcript_text = transcript_text
            
            # Step 2: Setup RAG
            status_text.text("🧠 Setting up RAG system...")
            progress_bar.progress(75)
            vectorstore, docs = setup_rag_system(transcript_text)
            
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.docs = docs
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                st.markdown("""
                <div class="status-success">
                    ✅ <strong>System Ready!</strong> You can now ask questions efficiently.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-error">❌ Failed to set up RAG system.</div>', unsafe_allow_html=True)

# Question Answering Interface
if "vectorstore" in st.session_state and "transcript_text" in st.session_state:
    st.markdown("---")
    st.markdown("## 💬 Ask Your Questions")
    
    # Question input
    user_question = st.text_input(
        "🤔 What would you like to know?", 
        placeholder="Ask anything about the video content...",
        key="question_input"
    )
    
    if user_question:
        # Processing section
        st.markdown("### 🔍 Processing Your Question")
        
        # Step 1: RAG Analysis
        with st.spinner("🔍 Searching transcript locally..."):
            rag_answer, confidence, context = get_rag_answer(user_question, st.session_state.vectorstore)
        
        # Display confidence analysis
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**📊 RAG Analysis Results:**")
            confidence_percentage = f"{confidence:.1%}"
            st.write(f"Confidence Score: **{confidence_percentage}**")
        
        with col2:
            if confidence > st.session_state.rag_confidence_threshold:
                st.markdown('<span class="confidence-high">🟢 High Confidence</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="confidence-low">🟡 Low Confidence</span>', unsafe_allow_html=True)
        
        with col3:
            threshold_percentage = f"{st.session_state.rag_confidence_threshold:.1%}"
            st.write(f"Threshold: {threshold_percentage}")
        
        # Answer Generation based on mode
        st.markdown("### 🎯 Answer")
        
        if "RAG Only" in answer_mode:
            # RAG only mode
            if rag_answer:
                st.markdown(f"""
                <div class="answer-box">
                    <h4>📚 RAG Answer:</h4>
                    <p>{rag_answer}</p>
                </div>
                """, unsafe_allow_html=True)
                st.info("✅ Answered locally - no API call needed!")
            else:
                st.markdown('<div class="status-warning">❌ No confident answer found in transcript. Try lowering the confidence threshold or use a different mode.</div>', unsafe_allow_html=True)
        
        elif "Smart Mode" in answer_mode:
            # Smart mode: RAG first, Groq if needed
            if rag_answer and confidence > st.session_state.rag_confidence_threshold:
                st.markdown(f"""
                <div class="answer-box">
                    <h4>📚 RAG Answer:</h4>
                    <p>{rag_answer}</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ Answered locally - no API call needed!")
            else:
                with st.spinner("🤖 Getting enhanced answer from Groq..."):
                    groq_answer = get_groq_answer(user_question, context, groq_api_key)
                
                st.markdown(f"""
                <div class="answer-box">
                    <h4>🚀 Enhanced Answer :</h4>
                    <p>{groq_answer}</p>
                </div>
                """, unsafe_allow_html=True)
                st.warning(f"⚡ Used  API call (Total: {st.session_state.groq_usage_count})")
        
        elif "Enhanced Mode" in answer_mode:
            # Always use Groq for enhanced answers
            with st.spinner("🤖 Getting enhanced answer..."):
                enhanced_answer = get_enhanced_answer(user_question, context, groq_api_key)
            
            st.markdown(f"""
            <div class="answer-box">
                <h4>🎯 Enhanced Answer (RAG + Groq):</h4>
                <p>{enhanced_answer}</p>
            </div>
            """, unsafe_allow_html=True)
            st.info(f"⚡ Used Groq API call (Total: {st.session_state.groq_usage_count})")
        
        # Context section
        with st.expander("📖 View Context Used", expanded=False):
            st.text_area(
                "Relevant transcript context:", 
                context[:1000] + "..." if len(context) > 1000 else context, 
                height=150,
                disabled=True
            )

# Transcript Preview
if "transcript_text" in st.session_state:
    st.markdown("---")
    with st.expander("📄 Full Transcript Preview", expanded=False):
        st.text_area(
            "Complete transcript:", 
            st.session_state.transcript_text, 
            height=300,
            disabled=True
        )

# Help and Tips Section
st.markdown("---")
with st.expander("❓ Usage Tips & Help", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 Smart Mode (Recommended):**
        - Tries RAG first (free, instant)
        - Only uses Groq if RAG confidence is low
        - Saves API calls automatically
        - Best for most use cases
        """)
    
    with col2:
        st.markdown("""
        **🚀 Enhanced Mode:**
        - Always uses Groq for best quality
        - More API calls but better answers
        - Good for complex questions
        - Comprehensive explanations
        """)
    
    with col3:
        st.markdown("""
        **💾 RAG Only Mode:**
        - No API calls ever
        - Fast but basic answers
        - Good for simple fact-finding
        - Cost-effective option
        """)
    
    st.markdown("---")
    st.markdown("""
    **💡 Tips to Save API Calls:**
    - Use Smart Mode for optimal balance
    - Increase confidence threshold to rely more on RAG
    - Ask specific questions about video content
    - Use RAG Only for simple queries
    - Review context to understand what information is available
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🧠 Smart RAG YouTube Tutor - Efficient AI-powered learning from video content</p>
</div>
""", unsafe_allow_html=True)
