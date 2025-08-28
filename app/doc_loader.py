import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
import pptx, docx
import retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import BaseLoader
import os

class CustomDocumentLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path
        end = self.file_path.split(".", 1) 
        self.ext = "." + end[1]

    def check(self):
        text = self.file_path
        delimiter = "."
        end = text.split(delimiter, 1) 

        print(f"Initialized loader for {self.ext} file: {'.' + end[1]}")



    def save_uploaded_file(uploaded_file):
        file_path = os.path.join("temp_files", uploaded_file.name)
        os.makedirs("temp_files", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path

    def lazy_load(self):
        if self.ext == ".txt":
            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    yield Document(page_content=line.strip(), metadata={"line": i+1, "source": self.file_path})
        elif self.ext == ".pdf":
            loader = PyPDFLoader(self.file_path, mode = "single", pages_delimiter="\n-------THIS IS A CUSTOM END OF PAGE-------\n",) #PDF can be extracted in single or multiple pages
            for doc in loader.load():
                yield doc
                print(f"Page Content: {doc.page_content[:100]}...") # Print first 100 characters
                print(f"Metadata: {doc.metadata}")
                print("-" * 30)
        elif self.ext == ".docx":
            loader = UnstructuredWordDocumentLoader(self.file_path)
            for doc in loader.load():
                yield doc
        elif self.ext == ".pptx":
            loader = UnstructuredPowerPointLoader(self.file_path, mode="elements")
            docs = loader.load()
            for doc in docs:
                yield doc

        else:
            raise ValueError(f"Unsupported file type: {self.ext}")



    def preprocess_file_content(query, file_content, state):

        chunks = retriever.chunck_text(file_content, chunk_size=1000, chunk_overlap=200)
        embeddings = retriever.embedd_chuncks(chunks)
        top_chunks = retriever.semantic_search_files(query, chunks, embeddings,90, top_k= 1000)
        answer = retriever.call_model_for_file(query, top_chunks=top_chunks, llm=retriever.llm)

        return answer


