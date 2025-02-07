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
import re
from googletrans import Translator
import pandas as pd

st.set_page_config(page_title="Chile-Chatbot", page_icon="assets/roche-logo.jpeg")

st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            background-color: #1482FA; 
            color: white;
        }

        section[data-testid="stSidebar"] select {
            color: black !important; 
            background-color: white !important; 
            border: 1px solid #ddd !important;
            border-radius: 4px !important;
            padding: 5px !important;
        }

        section[data-testid="stSidebar"] label {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Hide Streamlit's default header and footer
hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def language_translation(input_string,source_lan,target_lag):
    language = {'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy', 'azerbaijani': 'az', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny', 'chinese (simplified)': 'zh-cn', 'chinese (traditional)': 'zh-tw', 'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'filipino': 'tl', 'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka', 'german': 'de', 'greek': 'el', 'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'he', 'hindi': 'hi', 'hmong': 'hmn', 'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'indonesian': 'id', 'irish': 'ga', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'korean': 'ko', 'kurdish (kurmanji)': 'ku', 'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lithuanian': 'lt', 'luxembourgish': 'lb', 'macedonian': 'mk', 'malagasy': 'mg', 'malay': 'ms', 'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'mongolian': 'mn', 'myanmar (burmese)': 'my', 'nepali': 'ne', 'norwegian': 'no', 'odia': 'or', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', 'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm', 'scots gaelic': 'gd', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so', 'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta', 'telugu': 'te', 'thai': 'th', 'turkish': 'tr', 'ukrainian': 'uk', 'urdu': 'ur', 'uyghur': 'ug', 'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'}
    googletrans_translator = Translator()
    googletrans_result = googletrans_translator.translate(input_string,src= language[source_lan], dest= language[target_lag])
    return googletrans_result.text

# Add logo and styling
st.image("assets/final.png", width=500)

st.sidebar.title("Menu")

st.sidebar.subheader("Choose an Department:")
institution_options = [
    "Home",  # Adding Home option
    "Institute of Public Health",
    "Ministry of Health",
    "National Health Fund",
    "Superintendency of Health",
    "Supply Center of the National Health Services System"
]
selected_institution = st.sidebar.selectbox("Department", institution_options)

if selected_institution != "Home":

    st.sidebar.subheader("Choose a Year:")
    year_options = ["2023", "2024"]
    selected_year = st.sidebar.selectbox("Year", year_options)
    user_prefered_language = st.sidebar.radio("Select Language", ["english", "spanish"])

    st.session_state.year = selected_year

    # Combine institution and year for the final choice
    choice = f"{selected_institution} {selected_year}"
else:
    choice = "Home"

st.sidebar.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
st.sidebar.image("assets/yaali_animal.png", width=200)

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
                                   temperature=0.8, convert_system_message_to_human=True)
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
    - Before responding, analyze the context and input thoroughly to ensure the best answer is provided with 100% accuracy.
    - Use the retrieved context effectively to provide responses with the highest degree of accuracy, ensuring the answers are explanatory, interactive, and aligned with the user's needs.
    - Provide thorough and detailed answers.
    - Responses must be more detailed, thorough, and comprehensive, ensuring they address all aspects of the user's query effectively.
    - You must want to answer the question if the user query is somewhat related to the context below.
    - Remember all the context and chat history the user has provided and answer the question in natural language.
    - Pre-trained knowledge can only be used to support or clarify responses, but the final response must strictly rely on the provided context and chat history. Any information beyond the given context and chat history should not be included.
    - You should be a more interactive AI chatbot. Be engaging and ensure the conversation remains interactive and not boring.
    - Response should be professional and gentle; don't use offensive language.
    - Structure your response professionally in a **point-by-point**, **bold**, **italic**, or **bullet-point** format where appropriate.
    - If the user query is an open-ended question, act like a normal conversational chatbot.
    - Generate related questions based on the context below, ensuring that each generated question is relevant and can be answered using the provided context and chat history.

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

    # faiss_index = vector_store.index  # Get the underlying FAISS index
    # faiss_index.nprobe = 5  # Adjust nprobe for faster or more accurate retrieval

    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, prompt
    )

    question_answer_chain = create_stuff_documents_chain(model, prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def extract_meeting_details(text):
    meetings = []

    # Split the text by occurrences of 'On' to handle multiple meetings
    meeting_texts = re.split(r'(?=On \d{4}-\d{2}-\d{2} at)', text)

    for meeting_text in meeting_texts:
        details = {}

        # Extract date and time
        date_time_match = re.search(r'On (.*?) at (.*?),', meeting_text)
        if date_time_match:
            details['Date & Time'] = f"{date_time_match.group(1)} {date_time_match.group(2)}"

        # Extract official name and position
        official_match = re.search(r'at \d{1,2}:\d{2} [AP]M, (.*?) the (.*?), attended', meeting_text)
        if official_match:
            details['Official Name'] = official_match.group(1).rstrip(',')  # Remove trailing comma

        # Extract meeting platform
        platform_match = re.search(r'attended a (.*?)\. The', meeting_text, re.DOTALL)
        if platform_match:
            details['Platform'] = platform_match.group(1).strip()

        # Extract duration
        duration_match = re.search(r'The meeting lasted (.*?) and', meeting_text)
        if duration_match:
            details['Duration'] = duration_match.group(1)

        # Extract subjects discussed
        subjects_match = re.search(r'focused on the (.*?)\.', meeting_text)
        if subjects_match:
            details['Subjects Discussed'] = subjects_match.group(1)

        # Extract purpose
        purpose_match = re.search(r'\*\*Purpose:\*\*\n(.*?)\.', meeting_text)
        if purpose_match:
            details['Purpose'] = purpose_match.group(1)

        # Extract participants
        participants_match = re.search(r'\*\*Participants:\*\*\nThe meeting included:\n(.*?)\n\n', meeting_text, re.DOTALL)
        if participants_match:
            details['Participants'] = participants_match.group(1).replace('\n', ', ')

        # Extract key details
        identifier_match = re.search(r'Meeting Identifier: (.*?)\n', meeting_text)
        if identifier_match:
            details['Meeting Identifier'] = identifier_match.group(1)

        if details:
            meetings.append(details)

    return meetings

# Function to convert retrieved documents to DataFrame and display it in Streamlit
def display_meetings_as_table(docs):
    meetings_data = []

    for doc in docs:
        meetings = extract_meeting_details(doc.page_content)
        meetings_data.extend(meetings)

    # Create DataFrame
    df = pd.DataFrame(meetings_data)

    # Display DataFrame in a container with a height of 500
    container = st.container(border=True, height=500)
    if not df.empty:
        container.dataframe(df)
    else:
        container.warning("No meeting details found.")

if choice != "Home":
    db_name = f"faiss_index_{selected_institution.replace(' ', '_')}_{selected_year}"

    if choice != st.session_state.Department or selected_year != st.session_state.year:
        st.session_state.chat_history = []

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
                output_generated_text = response["answer"]
                
                if user_prefered_language != "english":
                    output_generated_text = language_translation(output_generated_text,"english",user_prefered_language)
                
                st.chat_message("assistant").markdown(output_generated_text)

            with st.expander("See relevant documents"):
                relevant_docs = get_more_relevant_docs(prompt, top_k=100)
                container = st.container(border=True, height=500)
                for idx, doc in enumerate(relevant_docs):
                    container.success(f"Meeting : {idx + 1}")
                    container.markdown(doc.page_content)
                    container.markdown("""

                                        """)
            with st.expander("See relevant raw data"):
                relevant_docs = get_more_relevant_docs(prompt, top_k=100)
                display_meetings_as_table(relevant_docs)

# add the institution to the session state
st.session_state.Department = choice