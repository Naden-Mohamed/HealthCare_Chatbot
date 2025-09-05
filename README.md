# Healthcare Chatbot system
***RAG (Retrieval-Augmented Generation) based application designed for medical question answering and document analysis. 
The system combines trusted medical source retrieval with LLM capabilities to provide accurate healthcare information responses.
This chatbot follows a layered architecture with clear separation between user interface (Streamlit), processing logic, and external service integrations. 
The system supports both general medical queries through web search and document-specific question answering through uploaded files***
<img width="1702" height="502" alt="image" src="https://github.com/user-attachments/assets/64b0cd04-15be-4e46-a541-5de53d7dae26" />


## Medical Information Retrieval
### The system provides healthcare information through multiple data sources:
+ Web Search: DuckDuckGo trusted medical sources	
+ Knowledge Base: Wikipedia API integration	
+ Document Analysis: Multi-format file processing [PDF, DOCX, PPTX, TXT]	
+ Semantic Search: FAISS vector similarity with credibility scoring
+ Processing Pipeline: File upload → Content extraction → Text chunking → Vector embedding → Semantic search

## Main RAG Pipeline

| Step | Function | Purpose |
| :---         |     :---:      | :---    |
| 1   | fetch_trusted_medical_webpages()     | Retrieve web content from trusted sources (DuckDuckGo Search Engine)    |
| 2   | fetch_clean_content()       | Extract clean text from web pages|
| 3   | chunck_text()     | Split content into manageable chunk |
| 4   | embedd_chuncks()       | Generate vector embeddings using HuggingFaceEmbeddings model("sentence-transformers/all-MiniLM-L6-v2")|
| 5   | semantic_search()     | Find most relevant chunks (L2 distance similarity)|
| 6   | call_model()       | Generate response using LLM (GroqAPI)|


## File Processing Workflow
The system supports multiple document formats through the CustomDocumentLoader class:
- Supported Formats: PDF, DOCX, PPTX, TXT
- Processing Pipeline: File upload → Content extraction → Text chunking → Vector embedding → Semantic search
- Storage: Temporary file storage in session state
- Query Processing: Context-aware responses using file content

## Conversational Memory Management
The system maintains conversation context through LangGraph-based state management:
* Message State: MessagesState tracks conversation history
* Memory Summarization: Automatic summarization when conversation exceeds 4 messages
* Workflow Orchestration: StateGraph manages conversation flow
* Persistent Storage: MemorySaver provides conversation checkpointing

## Session Management
Streamlit session state manages multiple conversation contexts:
- st.session_state.state: LangGraph MessagesState instance
- st.session_state.messages: Chat message history for UI display
- st.session_state.file_content: Processed document content
- st.session_state.uploaded_file: Current uploaded file reference

### Installation

* Clone the repository:

```
git clone https://github.com/your-username/healthcare-chatbot-rag.git
cd healthcare-chatbot-rag
```

* Create and activate a virtual environment:

```
python -m venv .venv 
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

* Install dependencies:

```
pip install -r requirements.txt
```

* Set up environment variables in .env:

```
GROQ_API_KEY=your_groq_api_key
```

* Usage

Run the chatbot:

``` 
python app/chatbot.py
```





