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
    "Supply Center of the National Health Services System"
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
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": top_k,  # number of documents to retrieve
                "score_threshold": 0.3  # only include docs that exceed this similarity
            }
        )
        # Retrieve documents based on similarity threshold and top_k
        docs = retriever.invoke(query)
        return docs
    except Exception as e:
        st.error(f"Error retrieving relevant documents: {e}")
        return []

def get_conversational_chain(vector_store):
    system_prompt = """
         Your name is AI Bot and you should also act like expert assistant and natural bot to answer all questions. 
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 

        IMPORTANT INSTRUCTIONS:
        - you should be more interactive ai chatbot , Be an engaging and conversation must be engage with user, it should not be like boring
        - Response should be professional and gentle, don't use offensive language.
        - Response should be structured , professional , Point by point wise , bold , italic , bullet point wise.
        - if the user query is an open-ended question and then you should act like a normal conversation chatbot.
        - You want to generate related questions based on the context below, ensuring that each generated question is relevant and can be answered using the provided context and chat history.
        - Remember all the context and chat history the user has provided and answer the question in natural language.
        - you must want to answer the question. if user query is somewhat related it self to the below context. yuo want to answer.
        - Provide thorough and detailed answers.
        - Pre-trained knowledge can only be used to support or clarify responses, but the final response must strictly rely on the provided context and chat history. Any information beyond the given context and chat history should not be included.
        - Responses must be more detailed, thorough, and comprehensive, ensuring they address all aspects of the user's query effectively.
        - Use the retrieved context effectively to provide responses with the highest degree of accuracy, ensuring the answers are explanatory, interactive, and aligned with the user's needs.

        Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        Use the following chat history and context to generate a helpful answer to the user’s question.

        Chat History:
        {chat_history} \n
        always you must want to give more detailed answer.
        Context: {context} \n
        Follow Up Input: {input} \n
        Helpful Answer:
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    faiss_index = vector_store.index  # Get the underlying FAISS index
    faiss_index.nprobe = 5  # Adjust nprobe for faster or more accurate retrieval

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 50}  # Only include 'k' here, no nprobe
    )

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
                relevant_docs = get_more_relevant_docs(prompt, top_k=100)
                for doc in relevant_docs:
                    st.write(doc)
