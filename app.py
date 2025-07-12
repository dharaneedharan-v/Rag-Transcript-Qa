# import os 
# import streamlit as st 
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter

# from pytube import YouTube
# from youtube_transcript_api import (
#     YouTubeTranscriptApi, 
#     TranscriptsDisabled, 
#     NoTranscriptFound, 
#     VideoUnavailable
# )

# from dotenv import load_dotenv
# load_dotenv()
# # st.set_page_config(page_title="YouTube Video Q&A", page_icon=":video_camera:")
# # st.title("YouTube Video Q&A :video_camera:")
# # st.write("Ask questions about a YouTube video and get answers using AI.")
# # st.write("Enter the YouTube video URL below:")
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# def get_transcript(url):
#     try:
#         video_id = YouTube(url).video_id
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
#         transcript_data = transcript_list.find_transcript(['en'])
#         text = " ".join([item.text for item in transcript_data])
#         return text
#     except TranscriptsDisabled:
#         st.error("Transcripts are disabled for this video.")
#     except NoTranscriptFound:
#         st.error("No transcript found for this video.")
#     except VideoUnavailable:
#         st.error("Video is unavailable.")
#     except CouldNotRetrieveTranscript:
#         st.error("Could not retrieve transcript.")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
    


# def Save_transcript_to_file(transcript, filename = "transcript.txt"):
#     with open(filename, "w" , encoding="utf-8") as f:
#         f.write(transcript)
#         st.success(f"Transcript saved to {filename}")

# st.title("Ask power Tutor ")
# st.write("Ask questions about a YouTube video and get answers using AI.")

# video_url = st.text_input("Enter YouTube video URL:")
# if st.button("Process Video"):
#     transcript_text = get_transcript(video_url)
#     if transcript_text:
#         Save_transcript_to_file(transcript_text)
#         st.success("Transcript retrieved successfully!")

#         loader   = TextLoader("transcript.txt", encoding="utf-8")
#         documents = loader.load()
#         spiltter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

#         embeddings = OpenAIEmbeddings()
#         vectorstore = FAISS.from_documents(documents, embeddings)
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=OpenAI(temperature=0),
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever()
#         )

#         st.session_state.qa_chain = qa_chain
#         st.success("Vector store and QA chain created successfully!")

#     else :
#         st.warning("No transcript available for the provided video URL.")
# if qa_chain in st.session_state:
#     user_question = st.text_input("Ask a question about the video:")
#     if user_question:
#         answer = st.session_state.qa_chain.run(user_question)
#         st.write("Answer:", answer)


# import os
# import streamlit as st
# from langchain_community.embeddings import 
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain_community.llms import OpenAI
# from pytube import YouTube
# from youtube_transcript_api import (
#     YouTubeTranscriptApi,
#     TranscriptsDisabled, 
#     NoTranscriptFound,
#     VideoUnavailable,                                                                                                        CouldNotRetrieveTranscript
# )
# from dotenv import load_dotenv
# load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# def get_youtube_transcript(url):
#     try:
#         video_id = YouTube(url).video_id
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
#         transcript = transcript_list.find_transcript(["en"])
#         transcript_data = transcript.fetch()
#         text = " ".join([item.text for item in transcript_data])
#         return text
#     except TranscriptsDisabled:
#         st.error("Transcripts are disabled for this video.")
#     except NoTranscriptFound:
#         st.error("No transcript found for this video.")
#     except VideoUnavailable:
#         st.error("This video is unavailable.")
#     except CouldNotRetrieveTranscript:
#         st.error("Could not retrieve transcript. It may not be available in your region.")
#     except Exception as e:
#         st.error(f"Unexpected error getting transcript: {e}")
#     return ""

# def save_transcript_to_file(text, filename="transcript.txt"):
#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(text)

# st.title("AI-Powered Tutor")
# st.write("Ask questions from YouTube lecture transcripts.")

# video_url = st.text_input("Enter YouTube Video URL")

# if st.button("Process Video"):
#     if video_url:
#         transcript_text = get_youtube_transcript(video_url)
#         if transcript_text:
#             save_transcript_to_file(transcript_text)

#             loader = TextLoader("transcript.txt", encoding="utf-8")
#             documents = loader.load()

#             splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             docs = splitter.split_documents(documents)

#             embeddings = OpenAIEmbeddings()
#             vectorstore = FAISS.from_documents(docs, embeddings)
#             retriever = vectorstore.as_retriever()
#             qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

#             st.session_state.qa_chain = qa_chain
#             st.success("Transcript processed successfully! You can now ask questions.")
#     else:
#         st.warning("Please enter a valid YouTube URL.")

# if "qa_chain" in st.session_state:
#     user_question = st.text_input("Ask a question from the video transcript")
#     if user_question:
#         answer = st.session_state.qa_chain.run(user_question)
#         st.write("**Answer:**", answer)

# import os
# import streamlit as st
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain_community.llms import HuggingFacePipeline
# from langchain_groq import ChatGroq
# from transformers import pipeline
# from pytube import YouTube
# from youtube_transcript_api import (
#     YouTubeTranscriptApi,
#     TranscriptsDisabled, 
#     NoTranscriptFound,
#     VideoUnavailable,                                                                                                       
#     CouldNotRetrieveTranscript
# )
# from dotenv import load_dotenv

# load_dotenv()

# # Option 1: Use free HuggingFace models (no API key needed)
# @st.cache_resource
# def load_free_llm():
#     """Load a free HuggingFace model for text generation"""
#     try:
#         # Using a smaller, faster model that works well for Q&A
#         text_generator = pipeline(
#             "text-generation",
#             model="microsoft/DialoGPT-medium",
#             max_length=512,
#             temperature=0.7,
#             do_sample=True,
#             pad_token_id=50256
#         )
#         return HuggingFacePipeline(pipeline=text_generator)
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# @st.cache_resource
# def load_free_embeddings():
#     """Load free HuggingFace embeddings"""
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# def get_youtube_transcript(url):
#     try:
#         video_id = YouTube(url).video_id
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
#         transcript = transcript_list.find_transcript(["en"])
#         transcript_data = transcript.fetch()
#         text = " ".join([item["text"] for item in transcript_data])
#         return text
#     except TranscriptsDisabled:
#         st.error("Transcripts are disabled for this video.")
#     except NoTranscriptFound:
#         st.error("No transcript found for this video.")
#     except VideoUnavailable:
#         st.error("This video is unavailable.")
#     except CouldNotRetrieveTranscript:
#         st.error("Could not retrieve transcript. It may not be available in your region.")
#     except Exception as e:
#         st.error(f"Unexpected error getting transcript: {e}")
#     return ""

# def save_transcript_to_file(text, filename="transcript.txt"):
#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(text)

# def simple_qa_without_llm(question, transcript_text):
#     """Simple keyword-based Q&A as fallback"""
#     question_lower = question.lower()
#     sentences = transcript_text.split('.')
    
#     # Find sentences containing question keywords
#     relevant_sentences = []
#     question_words = question_lower.split()
    
#     for sentence in sentences:
#         sentence_lower = sentence.lower()
#         if any(word in sentence_lower for word in question_words if len(word) > 2):
#             relevant_sentences.append(sentence.strip())
    
#     if relevant_sentences:
#         return " ".join(relevant_sentences[:3])  # Return top 3 relevant sentences
#     else:
#         return "Sorry, I couldn't find relevant information in the transcript for your question."

# st.title("AI-Powered YouTube Tutor")
# st.write("Ask questions from YouTube lecture transcripts using Groq's lightning-fast AI!")

# # Model selection
# model_option = st.selectbox(
#     "Choose your approach:",
#     ["Groq API (Fast & Cheap)", "Free HuggingFace Model", "Simple Keyword Search"]
# )

# video_url = st.text_input("Enter YouTube Video URL")

# if st.button("Process Video"):
#     if video_url:
#         with st.spinner("Getting transcript..."):
#             transcript_text = get_youtube_transcript(video_url)
        
#         if transcript_text:
#             save_transcript_to_file(transcript_text)
#             st.session_state.transcript_text = transcript_text
            
#             if model_option == "Groq API (Fast & Cheap)":
#                 groq_api_key = st.text_input("Enter your Groq API key:", type="password")
#                 if groq_api_key:
#                     with st.spinner("Processing with Groq..."):
#                         # Use free HuggingFace embeddings with Groq LLM
#                         embeddings = load_free_embeddings()
                        
#                         # Create Groq LLM
#                         llm = ChatGroq(
#                             groq_api_key=groq_api_key,
#                             model_name="llama3-70b-8192",  # Fast and powerful model
#                             temperature=0.1
#                         )
                        
#                         # Process documents
#                         loader = TextLoader("transcript.txt", encoding="utf-8")
#                         documents = loader.load()
                        
#                         splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                         docs = splitter.split_documents(documents)
                        
#                         # Create vector store
#                         vectorstore = FAISS.from_documents(docs, embeddings)
#                         retriever = vectorstore.as_retriever()
                        
#                         # Create QA chain
#                         qa_chain = RetrievalQA.from_chain_type(
#                             llm=llm,
#                             retriever=retriever,
#                             return_source_documents=True
#                         )
                        
#                         st.session_state.qa_chain = qa_chain
#                         st.session_state.model_type = "groq"
#                         st.success("Transcript processed with Groq! Lightning-fast responses ready.")
#                 else:
#                     st.warning("Please enter your Groq API key.")
            
#             elif model_option == "Free HuggingFace Model":
#                 with st.spinner("Loading free AI model..."):
#                     # Load free models
#                     embeddings = load_free_embeddings()
#                     llm = load_free_llm()
                    
#                     if llm:
#                         # Process documents
#                         loader = TextLoader("transcript.txt", encoding="utf-8")
#                         documents = loader.load()
                        
#                         splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                         docs = splitter.split_documents(documents)
                        
#                         # Create vector store
#                         vectorstore = FAISS.from_documents(docs, embeddings)
#                         retriever = vectorstore.as_retriever()
                        
#                         # Create QA chain
#                         qa_chain = RetrievalQA.from_chain_type(
#                             llm=llm,
#                             retriever=retriever,
#                             return_source_documents=True
#                         )
                        
#                         st.session_state.qa_chain = qa_chain
#                         st.session_state.model_type = "huggingface"
#                         st.success("Transcript processed with free AI model! You can now ask questions.")
#                     else:
#                         st.error("Failed to load AI model. Try the 'Simple Keyword Search' option.")
            
#             elif model_option == "Simple Keyword Search":
#                 st.session_state.model_type = "simple"
#                 st.success("Transcript processed! Using keyword-based search.")
#     else:
#         st.warning("Please enter a valid YouTube URL.")

# # Question answering section
# if "transcript_text" in st.session_state:
#     user_question = st.text_input("Ask a question from the video transcript:")
    
#     if user_question:
#         if st.session_state.get("model_type") == "simple":
#             # Simple keyword-based search
#             answer = simple_qa_without_llm(user_question, st.session_state.transcript_text)
#             st.write("**Answer:**", answer)
        
#         elif st.session_state.get("model_type") in ["huggingface", "groq"] and "qa_chain" in st.session_state:
#             # Use the QA chain
#             with st.spinner("Thinking..."):
#                 try:
#                     result = st.session_state.qa_chain({"query": user_question})
#                     st.write("**Answer:**", result["result"])
#                 except Exception as e:
#                     st.error(f"Error generating answer: {e}")
#                     # Fallback to simple search
#                     answer = simple_qa_without_llm(user_question, st.session_state.transcript_text)
#                     st.write("**Fallback Answer:**", answer)

# # Display transcript preview
# if "transcript_text" in st.session_state:
#     with st.expander("View Transcript Preview"):
#         st.write(st.session_state.transcript_text[:1000] + "...")



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

def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(["en"])
        transcript_data = transcript.fetch()
        
        text_parts = []
        for item in transcript_data:
            if hasattr(item, 'text'):
                text_parts.append(item.text)
            elif isinstance(item, dict) and 'text' in item:
                text_parts.append(item['text'])
            else:
                text_parts.append(str(item))
        
        return " ".join(text_parts)
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