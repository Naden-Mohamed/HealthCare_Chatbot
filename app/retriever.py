import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import asyncio
import wikipedia
os.environ["USER_AGENT"] = "my-healthcare-chatbot/1.0"


## Wikipedia API Wrapper
def fetch_wikipedia_pages(query, top_n=3):
    search_results = wikipedia.search(query, results=top_n)
    pages_content = []
    for title in search_results:
        try:
            page = wikipedia.page(title)
            pages_content.append({
                "title": title,
                "content": page.content,
                "url": page.url
            })
        except wikipedia.DisambiguationError as e:
            # Pick the first option or skip
            page = wikipedia.page(e.options[0])
            pages_content.append({
                "title": page.title,
                "content": page.content,
                "url": page.url
            })
        except Exception as e:
            continue
    return pages_content



# Example usage of WebBaseLoader to load trusted medical webpages

def fetch_trusted_medical_webpages(query, top_n=3):
    search = DuckDuckGoSearchRun()
    custom_tool = search.bind(
        name="trusted_medical_webpages",
        description="Search for trusted medical webpages using DuckDuckGo.",
        args_schema= DuckDuckGoSearchRun(query=query, num_results=top_n),
        filters={"site": "who.int OR cdc.gov OR nih.gov OR mayo.edu"},
        ),
    
    results = custom_tool.invoke(query=query, num_results=top_n)
    for result in results:
        print(f"Title: {result['title']}\nURL: {result['url']}\nContent Snippet: {result['snippet'][:500]}\n")
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
    words = RecursiveCharacterTextSplitter(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
# print(f"Split into {len(split_docs)} chunks")
# print(f"First chunk:\n{split_docs[0].page_content[:500].strip()}\n")


# STEP 3: Convert the chunks to a format suitable for the retriever (indexing , embeddings)
def embedd_chuncks(chuncks):
    embedding = []
    for chunk in chuncks:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            input = chunk).data[0].embedding
        
        embedding.append(embedding_model)
    return embedding
# print(f"Generated embeddings for {len(embiddings)} chunks")
# print(f"First embedding vector: {embiddings[0][:5]}...")  # Print first 5 dimensions of the first embedding


# STEP 4: Create a retriever from the embeddings (Store using FAISS)

db = FAISS.from_documents(split_docs, embedding_model)
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
    asyncio.run(retrieve_documents())
    query = "hyperthyroidism"   
    trusted_pages = fetch_trusted_medical_webpages(query, top_n=5)
    for page in trusted_pages:      
        page['credibility'] = 0.8  # Adding credibility field
        page['source'] = 'Trusted Medical Source'  # Adding source field
        print(f"Title: {page['title']}\nURL: {page['url']}\nContent Snippet: {page['snippet'][:500]}\n")

    wiki_pages = fetch_wikipedia_pages(query, top_n=5)
    for page in wiki_pages:
        page['credibility'] = 0.6  
        page['source'] = 'Wikipedia'  
        print(f"Title: {page['title']}\nURL: {page['url']}\nContent Snippet: {page['content'][:500]}\n")

