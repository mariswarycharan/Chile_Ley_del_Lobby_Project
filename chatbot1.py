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
from deep_translator import GoogleTranslator
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Chile-Chatbot", page_icon="assets/roche-logo.jpeg")

def get_last_updated(excel_file="update_log.xlsx", sheet_name = "Log"):
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        if "Last Updated" in df.columns and not df.empty:
            return str(df["Last Updated"].iloc[-1])
    except Exception:
        pass
    return None

last_updated = get_last_updated()
if last_updated:
    st.sidebar.markdown(f"**Last Updated:** {last_updated}")

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


#def language_translation(input_string, source_lan, target_lag):
#    language = {
#        'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy',
#        'azerbaijani': 'az', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bosnian': 'bs',
#        'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny',
#        'chinese (simplified)': 'zh-cn', 'chinese (traditional)': 'zh-tw', 'corsican': 'co',
#        'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dutch': 'nl', 'english': 'en',
#        'esperanto': 'eo', 'estonian': 'et', 'filipino': 'tl', 'finnish': 'fi', 'french': 'fr',
#        'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka', 'german': 'de', 'greek': 'el',
#        'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'he',
#        'hindi': 'hi', 'hmong': 'hmn', 'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig',
#        'indonesian': 'id', 'irish': 'ga', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw',
#        'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'korean': 'ko', 'kurdish (kurmanji)': 'ku',
#        'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lithuanian': 'lt',
#        'luxembourgish': 'lb', 'macedonian': 'mk', 'malagasy': 'mg', 'malay': 'ms', 'malayalam': 'ml',
#        'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'mongolian': 'mn', 'myanmar (burmese)': 'my',
#        'nepali': 'ne', 'norwegian': 'no', 'odia': 'or', 'pashto': 'ps', 'persian': 'fa',
#        'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', 'romanian': 'ro', 'russian': 'ru',
#        'samoan': 'sm', 'scots gaelic': 'gd', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn',
#        'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so',
#        'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg',
#        'tamil': 'ta', 'telugu': 'te', 'thai': 'th', 'turkish': 'tr', 'ukrainian': 'uk',
#        'urdu': 'ur', 'uyghur': 'ug', 'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy',
#        'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'
#    }

#    translated_text = GoogleTranslator(
#        source=language.get(source_lan, 'auto'),
#        target=language.get(target_lag, 'en')
#    ).translate(input_string)

#    return translated_text

# Add logo and styling
st.image("assets/final.png", width=500)

st.sidebar.title("Menu")

st.sidebar.subheader("Choose an Department:")
institutions = {
    "Home": "Home",
    "Instituto de Salud Pública": "Institute of Public Health",
    "Ministerio de Salud": "Ministry of Health",
    "Fondo Nacional de Salud": "National Health Fund",
    "Superintendencia de Salud": "Superintendency of Health",
    "Central de Abastecimiento del Sistema Nacional de Servicios de Salud": "Supply Center of the National Health Services System"
}

display_options = list(institutions.keys())
selected_institution = st.sidebar.selectbox("Department", display_options)

if selected_institution != "Home":
    st.sidebar.subheader("Choose a Year:")
    current_year_val = datetime.now().year
    year_options = [str(y) for y in range(2023,current_year_val + 1)]
    selected_year = st.sidebar.selectbox("Year", year_options)
    #user_prefered_language = st.sidebar.radio("Select Language", ["english", "spanish"])
    st.session_state.year = selected_year
    selected_institution_db = institutions[selected_institution]
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

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                   temperature=0.8, convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

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
        vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
        results = vector_store.similarity_search_with_score(query, k=top_k)
        retrieved_docs = [{"content": res.page_content, "score": score, "metadata": res.metadata} for res, score in results]
        return retrieved_docs
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
    - Las respuestas deben estar en español.

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


def extract_meeting_details(text: str) -> pd.DataFrame:
    meeting_pattern = re.compile(
        r"On\s+(.+?)\s+at\s+(.+?),\s+(.+?),\s+the\s+(.+?),\s+attended\s+a\s+(.+?)\s+meeting\s+via\s+(.+?)\.\s*"
        r"The meeting lasted\s+(.+?)\s+and\s+focused\s+on\s+the\s+(.+?)\.\s*"
        r"\*\*Purpose:\*\*\s*(.+?)\.\s*"
        r"\*\*Participants:\*\*\s*The meeting included:\s*(.+?)\s*"
        r"\*\*Key Details:\*\*\s*Meeting Identifier:\s*(.+?)\s*"
        r"Platform:\s*(.+?)\s*"
        r"Duration:\s*(.+?)\s*"
        r"Subjects Discussed:\s*(.+?)\s*"
        r"Source Link:\s*(.+)\s*",
        re.DOTALL
    )

    meetings = meeting_pattern.findall(text)
    data = []

    for meeting in meetings:
        (
            date_str, time_str,
            official_name, position,
            meeting_type, place,
            duration, subjects,
            purpose, participants,
            identifier, platform,
            duration_detail, subjects_detail,
            link
        ) = meeting

        try:
            dt_str = f"{date_str} {time_str}"
            date_obj = datetime.strptime(dt_str, "%B %d, %Y %I:%M %p")
            formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            formatted_date = f"{date_str} {time_str}"

        meeting_type = meeting_type.capitalize()

        participant_list = participants.strip().split("\n")

        first_participant = True
        for participant in participant_list:
            participant_details = participant.strip().split(" (")
            if len(participant_details) == 2:
                full_name = participant_details[0].strip()
                other_details = participant_details[1].replace(")", "").split(", ")
                quality, works_for, represents = (other_details + ["", "", ""])[:3]

                works_for = works_for.replace("working for ", "").strip() if works_for else ""
                represents = represents.replace("representing ", "").strip() if represents else ""
            else:
                full_name, quality, works_for, represents = participant, "", "", ""

            row = [
                official_name if first_participant else "",
                position if first_participant else "",
                link if first_participant else "",
                identifier if first_participant else "",
                formatted_date if first_participant else "",
                meeting_type if first_participant else "",
                place if first_participant else "",
                duration if first_participant else "",
                subjects if first_participant else "",
                purpose if first_participant else "",
                full_name,
                quality if quality != "nan" else "",
                works_for if works_for != "nan" else "",
                represents if represents != "nan" else "",
            ]
            data.append(row)
            first_participant = False

    columns = [
        "full_name", "Position", "link", "Identifier", "date", "Shape", "Place",
        "Duration", "Subjects_covered", "Specification", "Assistant_Full_name",
        "Assistant_Quality", "Assistant_Works_for", "Assistant_Represents"
    ]
    df = pd.DataFrame(data, columns=columns)
    df.replace({np.nan: "", "nan": ""}, inplace=True)
    return df


def display_meetings_as_table(docs: list):
    all_dataframes = []
    for doc in docs:
        extracted_df = extract_meeting_details(doc["content"])
        if not extracted_df.empty:
            extracted_df["Distance Score"] = f"{doc['score']:.3f}"
            all_dataframes.append(extracted_df)

    final_df = pd.concat(all_dataframes, ignore_index=True) if all_dataframes else pd.DataFrame()

    final_df.index = range(1, len(final_df) + 1)
    final_df = final_df.rename_axis("S.No.")

    container = st.container(border=True, height=500)
    with container:
        st.markdown("<br>", unsafe_allow_html=True)
        if not final_df.empty:
            st.dataframe(final_df, use_container_width=True)
        else:
            st.warning("No meeting details found.")
if choice != "Home":
    db_name = f"FAISS_DB/faiss_index_{selected_institution_db.replace(' ', '_')}_{selected_year}"

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

                #if user_prefered_language != "english":
                    #output_generated_text = language_translation(output_generated_text,"english",user_prefered_language)

                st.chat_message("assistant").markdown(output_generated_text)

            with st.expander("See relevant documents"):
                relevant_docs = get_more_relevant_docs(prompt, top_k=100)
                container = st.container(border=True, height=500)
                for idx, doc in enumerate(relevant_docs):
                    container.success(f"Meeting : {idx + 1}")
                    container.markdown(f"**Distance Score:** {doc['score']:.3f}")
                    container.markdown(doc["content"])
                    container.markdown("""

                                        """)
            with st.expander("See relevant raw data"):
                relevant_docs = get_more_relevant_docs(prompt, top_k=100)
                display_meetings_as_table(relevant_docs)

st.session_state.Department = choice