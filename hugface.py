


import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_groq import ChatGroq
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("Transformers not available. HuggingFace model option will be disabled.")
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled, 
    NoTranscriptFound,
    VideoUnavailable,                                                                                                       
    CouldNotRetrieveTranscript
)
from dotenv import load_dotenv

load_dotenv()

# Option 1: Use free HuggingFace models (no API key needed)
@st.cache_resource
def load_free_llm():
    """Load a free HuggingFace model for text generation"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        # Using a smaller, faster model that works well for Q&A
        text_generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            max_length=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=50256
        )
        return HuggingFacePipeline(pipeline=text_generator)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_free_embeddings():
    """Load free HuggingFace embeddings"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(["en"])
        transcript_data = transcript.fetch()
        
        # Handle both old and new transcript data formats
        text_parts = []
        for item in transcript_data:
            if hasattr(item, 'text'):
                text_parts.append(item.text)
            elif isinstance(item, dict) and 'text' in item:
                text_parts.append(item['text'])
            else:
                # Fallback for other formats
                text_parts.append(str(item))
        
        text = " ".join(text_parts)
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. It may not be available in your region.")
    except Exception as e:
        st.error(f"Unexpected error getting transcript: {e}")
    return ""

def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def simple_qa_without_llm(question, transcript_text):
    """Simple keyword-based Q&A as fallback"""
    question_lower = question.lower()
    sentences = transcript_text.split('.')
    
    # Find sentences containing question keywords
    relevant_sentences = []
    question_words = question_lower.split()
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(word in sentence_lower for word in question_words if len(word) > 2):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        return " ".join(relevant_sentences[:3])  # Return top 3 relevant sentences
    else:
        return "Sorry, I couldn't find relevant information in the transcript for your question."

st.title("AI-Powered YouTube Tutor")
st.write("Ask questions from YouTube lecture transcripts using Groq's lightning-fast AI!")

# Model selection
available_options = ["Groq API (Fast & Cheap)", "Simple Keyword Search"]
if TRANSFORMERS_AVAILABLE:
    available_options.insert(1, "Free HuggingFace Model")

model_option = st.selectbox(
    "Choose your approach:",
    available_options
)

video_url = st.text_input("Enter YouTube Video URL")

if st.button("Process Video"):
    if video_url:
        with st.spinner("Getting transcript..."):
            transcript_text = get_youtube_transcript(video_url)
        
        if transcript_text:
            save_transcript_to_file(transcript_text)
            st.session_state.transcript_text = transcript_text
            
            if model_option == "Groq API (Fast & Cheap)":
                groq_api_key = st.text_input("Enter your Groq API key:", type="password")
                if groq_api_key:
                    with st.spinner("Processing with Groq..."):
                        # Use free HuggingFace embeddings with Groq LLM
                        embeddings = load_free_embeddings()
                        
                        # Create Groq LLM
                        llm = ChatGroq(
                            groq_api_key=groq_api_key,
                            model_name="llama3-70b-8192",  # Fast and powerful model
                            temperature=0.1
                        )
                        
                        # Process documents
                        loader = TextLoader("transcript.txt", encoding="utf-8")
                        documents = loader.load()
                        
                        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        docs = splitter.split_documents(documents)
                        
                        # Create vector store
                        vectorstore = FAISS.from_documents(docs, embeddings)
                        retriever = vectorstore.as_retriever()
                        
                        # Create QA chain
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=retriever,
                            return_source_documents=True
                        )
                        
                        st.session_state.qa_chain = qa_chain
                        st.session_state.model_type = "groq"
                        st.success("Transcript processed with Groq! Lightning-fast responses ready.")
                else:
                    st.warning("Please enter your Groq API key.")
            
            elif model_option == "Free HuggingFace Model":
                if not TRANSFORMERS_AVAILABLE:
                    st.error("Transformers package not properly installed. Please use Groq API or Simple Keyword Search.")
                else:
                    with st.spinner("Loading free AI model..."):
                        # Load free models
                        embeddings = load_free_embeddings()
                        llm = load_free_llm()
                        
                        if llm:
                            # Process documents
                            loader = TextLoader("transcript.txt", encoding="utf-8")
                            documents = loader.load()
                            
                            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                            docs = splitter.split_documents(documents)
                            
                            # Create vector store
                            vectorstore = FAISS.from_documents(docs, embeddings)
                            retriever = vectorstore.as_retriever()
                            
                            # Create QA chain
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                retriever=retriever,
                                return_source_documents=True
                            )
                            
                            st.session_state.qa_chain = qa_chain
                            st.session_state.model_type = "huggingface"
                            st.success("Transcript processed with free AI model! You can now ask questions.")
                        else:
                            st.error("Failed to load AI model. Try the 'Simple Keyword Search' option.")
            
            elif model_option == "Simple Keyword Search":
                st.session_state.model_type = "simple"
                st.success("Transcript processed! Using keyword-based search.")
    else:
        st.warning("Please enter a valid YouTube URL.")

# Question answering section
if "transcript_text" in st.session_state:
    user_question = st.text_input("Ask a question from the video transcript:")
    
    if user_question:
        if st.session_state.get("model_type") == "simple":
            # Simple keyword-based search
            answer = simple_qa_without_llm(user_question, st.session_state.transcript_text)
            st.write("**Answer:**", answer)
        
        elif st.session_state.get("model_type") in ["huggingface", "groq"] and "qa_chain" in st.session_state:
            # Use the QA chain
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": user_question})
                    st.write("**Answer:**", result["result"])
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    # Fallback to simple search
                    answer = simple_qa_without_llm(user_question, st.session_state.transcript_text)
                    st.write("**Fallback Answer:**", answer)

# Display transcript preview
if "transcript_text" in st.session_state:
    with st.expander("View Transcript Preview"):
        st.write(st.session_state.transcript_text[:1000] + "...")