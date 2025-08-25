import requests
import bs4
import wikipedia
import faiss
import numpy as np
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from memory import call_model
from langgraph.graph import MessagesState


os.environ["USER_AGENT"] = "my-healthcare-chatbot/1.0"
# Initialize Groq LLM
llm = ChatGroq(
    model="llama3-8b-8192",  # Free LLaMA 3 model
    temperature=1.0,  # Adjust temperature for creativity
    max_tokens=100,  # Limit response length
)
# Initialize HuggingFace Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Initialize memory state
state = MessagesState(messages=[])


## Wikipedia API Wrapper
def fetch_wikipedia_pages(query, top_n=3):
    search_results = wikipedia.search(query, results=top_n)
    pages_content = []

    for title in search_results:
        try:
            page = wikipedia.page(title)
        except wikipedia.DisambiguationError as e:
            page = wikipedia.page(e.options[0])
        except Exception:
            continue
        
        pages_content.append({
            "title": page.title,
            "content": page.content,
            "url": page.url,
            "credibility": 0.6,
            "source": "Wikipedia"
        })

    return pages_content



# Example usage of WebBaseLoader to load trusted medical webpages

def fetch_trusted_medical_webpages(query, top_n=3):
    trusted_query = f"{query} "
    search = DuckDuckGoSearchRun()
    #site:who.int OR site:cdc.gov OR site:nih.gov OR site:mayo.edu
    raw_results = search.run(trusted_query)
    results = []
    
    for snippet in raw_results[:top_n]:
        page = {
            "title": snippet[:60],  # optional: take first 60 chars as a "title"
            "snippet": snippet,
            "credibility": 0.8,
            "source": "Trusted Medical Source"
        }
        results.append(page)
        #print(f"Title: {page['title']}\nSnippet: {page['snippet'][:500]}\n")
    
    return results

# Fetch and clean pages content
def fetch_clean_content(url):
    try:
        response = requests.get(url, timeout = 10)
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [" ".join([p.get_text() for p in paragraphs])]
        return text
    except:
        return ""


# Example usage of PyPDFLoader to load and process a PDF document

#STEP 1: Load a PDF document
pdf_path = r"C:/Users/start/OneDrive/Desktop/Algorithm_Task.pdf"
loader = PyPDFLoader(pdf_path, mode = "single", pages_delimiter="\n-------THIS IS A CUSTOM END OF PAGE-------\n",) #PDF can be extracted in single or multiple pages
docs = loader.load()
# print(f"Loaded {len(docs)} pages from {pdf_path}")

# STEP 2: Split the document into smaller chunks
def chunck_text(text, chunk_size=1000, chunk_overlap=200):
    words = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return words.split_text(text)
# print(f"Split into {len(split_docs)} chunks")
# print(f"First chunk:\n{split_docs[0].page_content[:500].strip()}\n")


# STEP 3: Convert the chunks to a format suitable for the retriever (indexing , embeddings)
def embedd_chuncks(chuncks):
    embedding = []
    for chunk in chuncks:
        vector = embedding_model.embed_query(chunk)
        embedding.append(vector)

    return embedding
# print(f"Generated embeddings for {len(embiddings)} chunks")
# print(f"First embedding vector: {embiddings[0][:5]}...")  # Print first 5 dimensions of the first embedding


# STEP 4: Create a retriever from the embeddings (Store using FAISS)
def semantic_search(query, chuncks, embeddings, credability_score, top_k = 5):
    query_embed = embedding_model.embed_query(query)

    # L2 distance (Euclidean distance) as the similarity metric.
    index = faiss.IndexFlatL2(len(query_embed)) # Dimension of the embeddings
    index.add(np.array(embeddings).astype('float32'))
    # This index will store all the document chunks as vectors 
    # so that we can quickly compute distances to the query vector.
    # FAISS requires a NumPy array of type float32

    D, I = index.search(np.array([query_embed]).astype('float32'), len(embeddings))
    # D - distances of each chunk from the query (shape [1, num_chunks]), I - indices of nearest neighbors

    # Final score = similarity * credibility
    results = []
    for pos, idx in enumerate(I[0]):
        similarity = 1 / (1 + D[0][pos])  # Convert distance to similarity
        # We want a similarity score between 0 and 1 (higher = better),
        # so 1 / (1 + distance) is a simple conversion.
        final_score = similarity * credability_score[idx]
        results.append((chuncks[idx] , final_score))
    results.sort(key=lambda x: x[1], reverse=True)
    return [chunck for chunck, final_score in results[:top_k]]


# def LLM_Answer_Generation(query, top_chunks):
#     context = "\n\n".join(top_chunks)
#     prompt = f"Answer the following question using only the provided context:\n\n{context}\n\nQuestion: {query}\nAnswer:"

#     # ChatGroq expects a list of messages
#     messages = [
#         SystemMessage(content="You are a helpful medical assistant."),
#         HumanMessage(content=prompt)
#     ]

#     response = llm(messages)  # Direct call
#     return response.content  # Extract text

def medical_query_rag(query, top_k = 5, top_n_chuncks = 5):
       # Check cache
    # cached_answer = cache_query(query)
    # if cached_answer:
    #     print("Using cached answer")
    #     return cached_answer

    search_results = fetch_trusted_medical_webpages(query, top_k)

    all_chunks = []
    credibility_scores = []
    for page in search_results:
        content = fetch_clean_content(page.get("url", ""))  # Fetch content only if URL exists
        if not content:
            content = page["snippet"]  # fallback to snippet
        chunks = chunck_text(content)
        all_chunks.extend(chunks)
        credibility_scores.extend([page['credibility']] * len(chunks))

    embeddings = embedd_chuncks(all_chunks)

    top_chunks = semantic_search(query, all_chunks, embeddings, credibility_scores, top_k= top_n_chuncks)

        # Call your model with top chunks
    answer_obj = call_model(state, query, top_chunks=top_chunks, llm = llm)

    # Extract the assistant's latest response
    answer = answer_obj["messages"][-1].content

    # 5. Cache answer
    #cache_query(query, answer)

    return answer




# Vector stores are usually run as a separate service that requires some IO operations, 
# and therefore they might be called asynchronously. 
# That gives performance benefits as you don't waste time waiting for responses from external services. 
# That might also be important if you work with an asynchronous framework, such as FastAPI.

async def retrieve_documents():
    query = "Comparison of the recursive and non-recursive algorithm"
    # Run blocking FAISS call in a thread executor
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(None, lambda: db.similarity_search(query, k=3))
    print(f"Retrieved {len(docs)} relevant chunks for query '{query}'")
    print(f"First chunk:\n{docs[0].page_content[:500].strip()}\n")




if __name__ == "__main__":
    # asyncio.run(retrieve_documents())
    query = "symptoms of hypothyroidism"
    answer = medical_query_rag(query, top_k=5, top_n_chuncks=5)
    print("Final Answer:\n", answer)
    

