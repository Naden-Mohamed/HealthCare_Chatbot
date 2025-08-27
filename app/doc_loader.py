import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
import pptx, docx
import retriever


class CustomDocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        end = self.file_path.split(".", 1) 
        self.ext = "." + end[1]

    def check(self):
        text = self.file_path
        delimiter = "."
        end = text.split(delimiter, 1) 

        # print(f"Initialized loader for {self.ext} file: {'.' + end[1]}")


    def lazy_load(self):
        if self.ext == ".txt":
            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    yield Document(page_content=line.strip(), metadata={"line": i+1, "source": self.file_path})
        elif self.ext == ".pdf":
            loader = PyPDFLoader(file_path, mode = "single", pages_delimiter="\n-------THIS IS A CUSTOM END OF PAGE-------\n",) #PDF can be extracted in single or multiple pages
            docs = loader.load()
            print(docs[0].page_content[:1000])
        elif self.ext == ".docx":
            loader = UnstructuredWordDocumentLoader(self.file_path)
            for doc in loader.load():
                yield doc
        elif self.ext == ".pptx":
            loader = UnstructuredPowerPointLoader(self.file_path, mode="elements")
            docs = loader.load()
            for doc in docs:
                print(f"Page Content: {doc.page_content[:100]}...") # Print first 100 characters
                print(f"Metadata: {doc.metadata}")
                print("-" * 30)
        else:
            raise ValueError(f"Unsupported file type: {self.ext}")

# Example usage
file_path = r"C:\Users\start\OneDrive\Documents\Graphics\Graphics Lecture 01.pptx"
loader = CustomDocumentLoader(file_path)
print("Checking file type...")
loader.check()

## Test out the lazy load interface
for doc in loader.lazy_load():
    print()
    print(type(doc))
    print(doc)

def preprocess_file_content():
