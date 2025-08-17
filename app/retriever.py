import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_core.retrievers import VectorStoreRetriever


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
print(f"Generated embeddings for {len(embiddings)} chunks")
print(f"First embedding vector: {embiddings[0][:5]}...")  # Print first 5 dimensions of the first embedding
# STEP 4: Create a retriever from the embeddings
