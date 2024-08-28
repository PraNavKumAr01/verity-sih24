from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time

vectorstore_index_name = "sih-legal-data"
pdf_path = "/Users/pranav/Documents/Projects/sih-24/Economic Business and Commercial Laws.pdf"
chunk_size = 1000
chunk_overlap = 200

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

pc = Pinecone(api_key='748b6399-8e0d-4786-9f7e-a2a2694ec938')  
spec = ServerlessSpec(cloud='aws', region='us-east-1')  
if vectorstore_index_name in pc.list_indexes().names():  
    pc.delete_index(vectorstore_index_name)  
pc.create_index(  
    vectorstore_index_name,  
    dimension=768,
    metric='dotproduct',  
    spec=spec  
)

loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

chunks = text_splitter.split_documents(documents)

store = PineconeVectorStore(
    index_name = vectorstore_index_name,
    embedding = embeddings,
    pinecone_api_key = "748b6399-8e0d-4786-9f7e-a2a2694ec938"
)

print("Starting embedding process...")
start_time = time.time()
store.add_documents(chunks)
end_time = time.time()
total_time = end_time - start_time
print(f"Processed {len(chunks)} chunks from the PDF and stored them in the database in {total_time}.")