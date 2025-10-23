import os, glob
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

def build_rag_index(data_dir: str, index_path: str):
    pdfs = glob.glob(os.path.join(data_dir, "*.pdf"))
    docs = []
    for p in pdfs:
        loader = PyPDFLoader(p)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    vs.save_local(index_path)
    return index_path

def load_rag_index(index_path: str) -> FAISS:
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def rag_search(index_path: str, query: str, k: int = 5) -> List[str]:
    vs = load_rag_index(index_path)
    docs = vs.similarity_search(query, k=k)
    return [f"{d.metadata.get('source','')}: {d.page_content[:500]}" for d in docs]
