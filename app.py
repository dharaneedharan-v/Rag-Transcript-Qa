import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled, 
    NoTranscriptFound,
    VideoUnavailable,                                                                                                       
    CouldNotRetrieveTranscript
)
from dotenv import load_dotenv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

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
#         ytt_api = YouTubeTranscriptApi()
#         transcript_list = ytt_api.list(video_id)
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

def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        # Directly get the transcript in English
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

        text_parts = [item['text'] for item in transcript_data if 'text' in item]
        return " ".join(text_parts)

    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("The video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve the transcript.")
    except Exception as e:
        st.error(f"Error getting transcript: {e}")
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

# groq_api_key = st.text_input("🔑 Groq API Key:", type="password") or os.getenv("GROQ_API_KEY")
# if not groq_api_key:
#     st.error("🔑 Please provide a valid Groq API key either via input or .env file!")
#     st.stop()
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
        st.write(f"Debug: Using API key (first 5 chars): {groq_api_key[:5]}...")
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

# Streamlit UI
st.title("🧠 Smart RAG + Groq YouTube Tutor")
st.write("Efficient AI tutoring that saves API calls by using local search first!")

# Settings in sidebar
st.sidebar.header("⚙️ Settings")
st.session_state.rag_confidence_threshold = st.sidebar.slider(
    "RAG Confidence Threshold", 
    0.0, 1.0, 0.3, 0.1,
    help="Higher = More likely to use Groq. Lower = More RAG-only answers"
)

answer_mode = st.sidebar.radio(
    "Answer Mode:",
    ["🔍 Smart Mode (RAG → Groq if needed)", "🚀 Enhanced Mode (RAG + Groq)", "💾 RAG Only"]
)

# Display usage stats
st.sidebar.metric("Groq API Calls Used", st.session_state.groq_usage_count)

# Main interface
col1, col2 = st.columns([2, 1])

# with col1:
#     groq_api_key = st.text_input("🔑 Groq API Key:", type="password")
#     video_url = st.text_input("🎥 YouTube Video URL:")
with col1:
    video_url = st.text_input("🎥 YouTube Video URL:", key="video_url_input")


with col2:
    st.markdown("### 🎯 How it works:")
    st.markdown("1. **RAG First**: Search transcript locally")
    st.markdown("2. **Smart Decision**: Use Groq only if needed")
    st.markdown("3. **Save API calls**: Efficient resource usage")

if st.button("🚀 Process Video"):
    if not video_url:
        st.error("Please enter a YouTube URL!")
    else:
        with st.spinner("📝 Getting transcript..."):
            transcript_text = get_youtube_transcript(video_url)
        
        if transcript_text:
            st.session_state.transcript_text = transcript_text
            
            with st.spinner("🧠 Setting up RAG system..."):
                vectorstore, docs = setup_rag_system(transcript_text)
                
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.docs = docs
                    st.success("✅ System ready! Now you can ask questions efficiently.")
                else:
                    st.error("❌ Failed to set up RAG system.")

# Question answering
if "vectorstore" in st.session_state and "transcript_text" in st.session_state:
    st.markdown("---")
    st.markdown("### 💬 Ask Your Questions")
    
    user_question = st.text_input("🤔 What would you like to know?")
    
    if user_question:
        st.markdown("### 🔍 Processing...")
        
        # Step 1: Try RAG first
        with st.spinner("🔍 Searching transcript locally..."):
            rag_answer, confidence, context = get_rag_answer(user_question, st.session_state.vectorstore)
        
        # Display RAG results
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**📊 RAG Analysis:**")
            st.write(f"Confidence Score: {confidence:.2f}")
        with col2:
            confidence_color = "🟢" if confidence > st.session_state.rag_confidence_threshold else "🟡"
            st.write(f"{confidence_color} {'High' if confidence > st.session_state.rag_confidence_threshold else 'Low'} Confidence")
        
        # Decision logic based on mode
        if answer_mode == "💾 RAG Only":
            # RAG only mode
            if rag_answer:
                st.markdown("### 🎯 **Answer (RAG Only):**")
                st.write(rag_answer)
            else:
                st.markdown("### ❌ **No confident answer found in transcript**")
                st.write("Try lowering the confidence threshold or use a different mode.")
        
        elif answer_mode == "🔍 Smart Mode (RAG → Groq if needed)":
            # Smart mode: RAG first, Groq if needed
            if rag_answer and confidence > st.session_state.rag_confidence_threshold:
                st.markdown("### 🎯 **Answer (RAG):**")
                st.write(rag_answer)
                st.info("✅ Answered locally - no API call needed!")
            else:
                if groq_api_key:
                    st.markdown("### 🚀 **Enhanced Answer (Groq):**")
                    with st.spinner("🤖 Getting enhanced answer from Groq..."):
                        groq_answer = get_groq_answer(user_question, context, groq_api_key)
                        st.write(groq_answer)
                    st.warning(f"⚡ Used Groq API call (Total: {st.session_state.groq_usage_count})")
                else:
                    st.error("🔑 Need Groq API key for enhanced answers!")
        
        elif answer_mode == "🚀 Enhanced Mode (RAG + Groq)":
            # Always use Groq for enhanced answers
            if groq_api_key:
                st.markdown("### 🎯 **Enhanced Answer (RAG + Groq):**")
                with st.spinner("🤖 Getting enhanced answer..."):
                    enhanced_answer = get_enhanced_answer(user_question, context, groq_api_key)
                    st.write(enhanced_answer)
                st.info(f"⚡ Used Groq API call (Total: {st.session_state.groq_usage_count})")
            else:
                st.error("🔑 Need Groq API key for enhanced mode!")
        
        # Show context used
        with st.expander("📖 Context Used"):
            st.text_area("Relevant transcript context:", context[:1000] + "...", height=150)

# Transcript preview
if "transcript_text" in st.session_state:
    with st.expander("📄 Full Transcript"):
        st.text_area("Complete transcript:", st.session_state.transcript_text, height=300)

# Help section
with st.expander("❓ How to Use Efficiently"):
    st.markdown("""
    **🎯 Smart Mode (Recommended):**
    - Tries RAG first (free, instant)
    - Only uses Groq if RAG confidence is low
    - Saves API calls automatically
    
    **🚀 Enhanced Mode:**
    - Always uses Groq for best quality
    - More API calls but better answers
    - Good for complex questions
    
    **💾 RAG Only Mode:**
    - No API calls ever
    - Fast but basic answers
    - Good for simple fact-finding
    
    **💡 Tips to Save API Calls:**
    - Use Smart Mode
    - Increase confidence threshold
    - Ask specific questions about video content
    - Use RAG Only for simple queries
    """)