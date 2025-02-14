from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def get_meeting_chunks(text):
    meetings = text.split("\n\n---\n")  # Splitting on "---" as it indicates a new meeting
    return [meeting.strip() for meeting in meetings if meeting.strip()]


def get_vector_store(meeting_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(meeting_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_Supply_Center_of_the_National_Health_Services_system_2023")


if __name__ == "__main__":
    file_path = "enhanced_meeting_details_story(Supply Center of the National Health Services system Translated).txt"
    text = get_pdf_text(file_path)
    print("Reading text from file completed successfully......................")

    meetings = get_meeting_chunks(text)
    print(f"{len(meetings)} individual meeting chunks created successfully......................")

    get_vector_store(meetings)
    print("Vector store created successfully......................")