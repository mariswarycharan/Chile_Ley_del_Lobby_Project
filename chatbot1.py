import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.sidebar.title("Menu")
options = [
    "Home",
    "Institute of Public Health",
    "Ministry of Health",
    "National Health Fund",
    "Superintendency of Health",
    "Supply Center of the National Health Services system"
]
choice = st.sidebar.selectbox("Choose a section:", options)

if choice == "Home":
    st.title("Welcome to the AI Chatbot Application")
    st.write("""This application allows you to interact with an AI-powered chatbot.
            Use the navigation menu to select a specific database or return to this homepage.
            """)

st.cache_resource(show_spinner=False)
def load_model():
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0 , convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    return model, embeddings

model, embeddings = load_model()

def load_database(db_name):
    try:
        vector_store = FAISS.load_local(
            db_name, embeddings, allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Error loading database {db_name}: {e}")
        return None

def get_more_relevant_docs(query, top_k):
    try:
        vector_store = FAISS.load_local(
            db_name, embeddings, allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
        docs = retriever.invoke(query)
        return docs
    except Exception as e:
        st.error(f"Error retrieving relevant documents: {e}")
        return []

def get_conversational_chain(vector_store):
    system_prompt = """
You are AI ChatBot, an expert assistant specializing in providing detailed and accurate responses strictly based on retrieved information. Your goal is to deliver factual, concise, and helpful answers without introducing speculative or fabricated content.

*INSTRUCTIONS:*
- Base all responses exclusively on the provided context. If the information is not available, clearly state that you do not have enough data to answer.
- Avoid generating information that is not explicitly stated or implied by the retrieved documents.
- Respond politely and informatively.
- Use headings, bullet points, and concise paragraphs for clarity and readability.
- Highlight key points, participants, and outcomes. Avoid over-explaining or speculating beyond the given data.
- Emphasize important actions, follow-ups, and next steps from meetings or discussions.

*ONGOING CONVERSATION:*
The following is a record of the conversation so far, including user queries and assistant responses. Use this to maintain context and provide answers in continuity with previous exchanges.

{chat_history}

*DOCUMENT CONTEXT (if available):*
The following context is retrieved from relevant documents related to the query.

{context}

*USER QUERY:*
{input}

*ASSISTANT RESPONSE:*
Provide a detailed response, keeping prior exchanges in mind. Refer to past questions and answers for continuity. Avoid repeating information unnecessarily but expand on new aspects related to the user's follow-up query.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6})

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, prompt
    )

    question_answer_chain = create_stuff_documents_chain(model, prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

if choice != "Home":
    db_name = f"faiss_index_{choice.replace(' ', '_')}"
    vector_store = load_database(db_name)

    if vector_store:
        chain = get_conversational_chain(vector_store)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        def user_input(user_question):
            response = chain.invoke(
                {
                    "input": user_question,
                    "context": "Your relevant context goes here",
                    "chat_history": st.session_state.chat_history,
                }
            )
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=response["answer"]))
            return response

        st.title(f"AI Chatbot - {choice}")

        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.chat_message("user").markdown(message.content)
            elif isinstance(message, AIMessage):
                st.chat_message("assistant").markdown(message.content)

        if prompt := st.chat_input("Ask your question here..."):
            st.chat_message("user").markdown(prompt)
            with st.spinner("Generating response..."):
                response = user_input(prompt)
                st.chat_message("assistant").markdown(response["answer"])

            with st.expander("See relevant documents"):
                relevant_docs = get_more_relevant_docs(prompt, top_k=50)
                for doc in relevant_docs:
                    st.write(doc)
