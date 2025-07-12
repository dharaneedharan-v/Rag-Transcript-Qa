# Smart RAG YouTube Tutor

![Project Logo](https://img.shields.io/badge/Smart%20RAG-YouTube%20Tutor-blueviolet?style=for-the-badge)

- A Streamlit-based web application that combines Retrieval-Augmented Generation (RAG) with YouTube video transcripts to provide efficient, AI-powered tutoring. This project leverages local search to minimize API calls, offering both cost-effective and high-quality answers to user queries about video content.


## Features

- **YouTube Transcript Processing**: Extracts and processes transcripts from YouTube videos using `youtube_transcript_api`.
- **RAG-Powered Search**: Uses FAISS and HuggingFace embeddings for fast, local retrieval of relevant transcript sections.
- **Smart Answer Modes**:
  - **Smart Mode**: Prioritizes local RAG search, falling back to Groq API for low-confidence answers.
  - **Enhanced Mode**: Combines RAG with Groq for detailed, high-quality responses.
  - **RAG Only Mode**: Uses only local search for cost-free, instant answers.
- **Confidence Scoring**: Calculates answer reliability using TF-IDF and word overlap metrics.
- **Customizable UI**: Streamlit-based interface with responsive design and custom CSS styling.
- **API Call Optimization**: Reduces dependency on external APIs to save costs and improve speed.
- **Transcript Preview**: View full or relevant transcript sections for transparency.

## Tech Stack

- **Python**: Core programming language.
- **Streamlit**: For building the interactive web interface.
- **LangChain**: For RAG implementation, document splitting, and embeddings.
- **HuggingFace Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` for text embeddings.
- **FAISS**: Efficient vector store for similarity search.
- **Groq API**: For enhanced answer generation (optional, based on mode).
- **YouTubeTranscriptApi**: For extracting YouTube video transcripts.
- **Pytube**: For YouTube video metadata extraction.
- **Scikit-learn**: For TF-IDF and cosine similarity calculations.
- **python-dotenv**: For managing API keys.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/smart-rag-youtube-tutor.git
   cd smart-rag-youtube-tutor
   ```

2. **Set Up a Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**: Create a `.env` file in the project root and add your Groq API key:

   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the Application**:

   ```bash
   streamlit run app.py
   ```

## Usage

1. **Access the App**: Open your browser and navigate to `http://localhost:8501` after running the app.

2. **Input a YouTube URL**: Enter a YouTube video URL in the provided text box and click "Process Video" to extract and process the transcript.

3. **Configure Settings** (in the sidebar):

   - Adjust the **RAG Confidence Threshold** to control when Groq API is used.
   - Select an **Answer Mode**:
     - **Smart Mode**: Balances speed and quality.
     - **Enhanced Mode**: Uses Groq for detailed answers.
     - **RAG Only Mode**: Uses local search only, no API calls.

4. **Ask Questions**: Enter your question about the video content in the question input box. The app will provide answers based on the selected mode.

5. **Review Results**:

   - View the answer, confidence score, and relevant transcript context.
   - Check the full transcript under the "Transcript Preview" section.
   - Monitor Groq API usage in the sidebar.

## How It Works

1. **Transcript Extraction**:

   - The app uses `pytube` and `youtube_transcript_api` to fetch the video transcript.
   - The transcript is saved locally for processing.

2. **RAG Setup**:

   - The transcript is split into chunks using `CharacterTextSplitter`.
   - HuggingFace embeddings are generated and stored in a FAISS vector store for fast retrieval.

3. **Question Answering**:

   - **RAG Search**: Queries are matched against the transcript using FAISS to retrieve relevant chunks.
   - **Confidence Scoring**: TF-IDF and word overlap metrics determine answer reliability.
   - **Answer Generation**:
     - In **Smart Mode**, high-confidence RAG answers are used; otherwise, Groq is invoked.
     - In **Enhanced Mode**, Groq generates detailed answers with RAG context.
     - In **RAG Only Mode**, only local RAG answers are provided.

4. **Optimization**:

   - Local RAG minimizes Groq API calls, reducing costs.
   - Confidence thresholds ensure Groq is only used when necessary.

## Configuration

- **RAG Confidence Threshold**:

  - Range: 0.0 to 1.0 (default: 0.3).
  - Higher values increase reliance on RAG, reducing API calls.
  - Lower values trigger Groq more often for better answers.

- **Answer Modes**:

  - **Smart Mode**: Ideal for most use cases, balancing speed and quality.
  - **Enhanced Mode**: Best for complex questions requiring detailed explanations.
  - **RAG Only Mode**: Cost-free, suitable for simple queries.

- **Custom Styling**:

  - The app uses custom CSS for a modern, responsive UI with gradient headers, feature cards, and status indicators.
