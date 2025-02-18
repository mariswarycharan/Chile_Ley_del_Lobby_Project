import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def get_meeting_chunks(text):
    meetings = text.split("\n\n---\n")
    return [meeting.strip() for meeting in meetings if meeting.strip()]

def get_vector_store(meeting_chunks, index_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(meeting_chunks, embedding=embeddings)
    vector_store.save_local(index_name)


