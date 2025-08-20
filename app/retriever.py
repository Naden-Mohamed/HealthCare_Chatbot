import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import asyncio
os.environ["USER_AGENT"] = "my-healthcare-chatbot/1.0"

## Wikipedia API Wrapper
import wikipedia

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

query = "hyperthyroidism"
wiki_pages = fetch_wikipedia_pages(query, top_n=5)
for page in wiki_pages:
    page['credibility'] = 0.8  # Adding credibility field
    page['source'] = 'Wikipedia'  # Adding source field
    print(f"Title: {page['title']}\nURL: {page['url']}\nContent Snippet: {page['content'][:500]}\n")



# Example usage of WebBaseLoader to load trusted medical webpages

# page_url = "https://www.who.int/news-room/fact-sheets/detail/anemia"

# loader = WebBaseLoader(web_paths=[page_url])
# import asyncio

# docs = []

# async def load_documents():
#     async for doc in loader.alazy_load():
#         docs.append(doc)

# asyncio.run(load_documents())

# assert len(docs) == 1
# doc = docs[0]
# print(f"{doc.metadata}\n")
# print(doc.page_content[:500].strip())



# Example usage of PyPDFLoader to load and process a PDF document

#STEP 1: Load a PDF document
pdf_path = r"C:/Users/start/OneDrive/Desktop/Algorithm_Task.pdf"
loader = PyPDFLoader(pdf_path, mode = "single", pages_delimiter="\n-------THIS IS A CUSTOM END OF PAGE-------\n",) #PDF can be extracted in single or multiple pages
docs = loader.load()
# print(f"Loaded {len(docs)} pages from {pdf_path}")

# STEP 2: Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)
# print(f"Split into {len(split_docs)} chunks")
# print(f"First chunk:\n{split_docs[0].page_content[:500].strip()}\n")

# STEP 3: Convert the chunks to a format suitable for the retriever (indexing , embeddings)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embiddings = embedding_model.embed_documents([doc.page_content for doc in split_docs])
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
