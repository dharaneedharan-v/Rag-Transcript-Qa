
import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
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

load_dotenv()

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

st.title("🚀 YouTube AI Tutor with Groq")
st.write("Lightning-fast Q&A from YouTube lecture transcripts!")

# Get Groq API key
groq_api_key = st.text_input("Enter your Groq API key:", type="password")
video_url = st.text_input("Enter YouTube Video URL:")

if st.button("🎯 Process Video"):
    if not groq_api_key:
        st.error("Please enter your Groq API key first!")
    elif not video_url:
        st.error("Please enter a YouTube URL!")
    else:
        with st.spinner("📝 Getting transcript..."):
            transcript_text = get_youtube_transcript(video_url)
        
        if transcript_text:
            save_transcript_to_file(transcript_text)
            st.session_state.transcript_text = transcript_text
            
            with st.spinner("🧠 Processing with Groq AI..."):
                try:
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
                    st.session_state.groq_api_key = groq_api_key
                    st.success("🎉 Transcript processed! Ready for lightning-fast Q&A!")
                    
                except Exception as e:
                    st.error(f"Error setting up Groq: {e}")
                    st.info("Falling back to simple keyword search...")
                    st.session_state.use_simple = True

# Question answering section
if "transcript_text" in st.session_state:
    st.markdown("---")
    st.markdown("### 💬 Ask Your Questions")
    
    user_question = st.text_input("🤔 What would you like to know about the video?")
    
    if user_question:
        if "qa_chain" in st.session_state and not st.session_state.get("use_simple", False):
            # Use Groq AI
            with st.spinner("🔍 Groq is thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": user_question})
                    st.markdown("### 🎯 **Answer:**")
                    st.write(result["result"])
                    
                    # Show source documents if available
                    if "source_documents" in result and result["source_documents"]:
                        with st.expander("📚 Source Context"):
                            for i, doc in enumerate(result["source_documents"][:2]):
                                st.write(f"**Context {i+1}:** {doc.page_content[:300]}...")
                                
                except Exception as e:
                    st.error(f"Error with Groq: {e}")
                    # Fallback to simple search
                    answer = simple_qa_without_llm(user_question, st.session_state.transcript_text)
                    st.markdown("### 🔍 **Fallback Answer:**")
                    st.write(answer)
        else:
            # Simple keyword-based search
            answer = simple_qa_without_llm(user_question, st.session_state.transcript_text)
            st.markdown("### 🔍 **Answer:**")
            st.write(answer)

# Display transcript preview
if "transcript_text" in st.session_state:
    with st.expander("📄 View Full Transcript"):
        st.text_area("Transcript:", st.session_state.transcript_text, height=300)

# Sidebar info
st.sidebar.markdown("### 🎓 How to Use:")
st.sidebar.markdown("1. Get your Groq API key from [console.groq.com](https://console.groq.com)")
st.sidebar.markdown("2. Enter your API key above")
st.sidebar.markdown("3. Paste any YouTube URL")
st.sidebar.markdown("4. Click 'Process Video'")
st.sidebar.markdown("5. Ask questions about the content!")

st.sidebar.markdown("### 💡 Tips:")
st.sidebar.markdown("- Works best with educational videos")
st.sidebar.markdown("- Try specific questions for better answers")
st.sidebar.markdown("- Groq is super fast and cost-effective!")
