from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.schema import Document
from dotenv import load_dotenv
import os
import sys
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = "YOUR GROQ_API_KEY"

DB_FAISS_PATH = "vectorstore/db_faiss"

# Load the CSV file using pandas
df = pd.read_csv("data/WHR_2019.csv")
print(f"CSV file loaded. Number of rows: {len(df)}")

# Create documents from the data
documents = [
    Document(
        page_content=f"Row {i}: " + " | ".join([f"{k}: {v}" for k, v in row.items()]),
        metadata={"row": i},
    )
    for i, row in enumerate(df.to_dict(orient="records"))
]

# Add a document with the total number of rows
documents.append(
    Document(
        page_content=f"The CSV file contains a total of {len(df)} rows.",
        metadata={"row": "total"},
    )
)

print(f"Number of documents created: {len(documents)}")

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(documents)
print(f"Number of text chunks: {len(text_chunks)}")

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert the text chunks into embeddings and save them into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)
docsearch.save_local(DB_FAISS_PATH)
print("Embeddings saved to FAISS")

# Initialize the Groq LLM
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.1,
    max_tokens=1024,
    top_p=1,
)

# Create the conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

# Main conversation loop
chat_history = []
while True:
    query = input("Input Prompt: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("Exiting")
        sys.exit()
    if query == "":
        continue

    result = qa({"question": query, "chat_history": chat_history})
    print("Response:", result["answer"])
    chat_history.append((query, result["answer"]))
