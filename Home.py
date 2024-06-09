import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import pickle

st.set_page_config(page_title="Brainiac Buddy", page_icon="🧠")
st.title("Brainiac Buddy :robot_face:")
st.markdown(
    """
    ## Brainiac Buddy: Unleash Your Inner Genius
    **Stop struggling with study material!** Brainiac Buddy is your personalized AI tutor that turns PDFs into powerful learning tools.  
    **Here's how it works:**

    * **Upload any PDF:**  Brainiac Buddy analyzes your study materials.
    * **Unlock Key Concepts:**  Our advanced AI identifies the most important information using RAG (Read, Analyze, Generate).
    * **Master the Material:**  Brainiac Buddy generates targeted questions to test your understanding and solidify your knowledge.

    **Brainiac Buddy is your secret weapon for academic success. Get ready to ace your next exam!**    
    """
)

store = {}
# Setup LLM
llm = ChatNVIDIA(
    model='meta/llama3-70b-instruct', nvidia_api_key='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'
)

@st.cache_resource(ttl="1h")
def processing_documents(uploaded_files):
    
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = NVIDIAEmbeddings(nvidia_api_key="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")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr")

    contextualize_q_system_prompt = (
        "You are a digital buddy for question-answering tasks. "
        "You help students to retrieve information and learn faster. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. "
        "Tone: Realistic, informative, and concise."
        "\n\n"
        "The question is:"
        "\n\n"
        "{input}"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # __________________________________________________________________________________________ #

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{context}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # __________________________________________________________________________________________ #

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store["this-is-a-unique-id"] = ChatMessageHistory()
        return store["this-is-a-unique-id"]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True, 
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

conversational_rag_chain = processing_documents(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        response = conversational_rag_chain.invoke(
            {"input": user_query},
            config={
                "configurable": {"session_id": "this-is-a-unique-id"},
            }
        )
        st.write(response['answer'])
        msgs.add_user_message(user_query)
        msgs.add_ai_message(response['answer'])

with st.sidebar:
    st.markdown("<hr><h2>Save Chat</h2>", unsafe_allow_html=True)
    name = st.text_input("Name for this chat")
    save = st.button("Save Chat")
    
    if save and not name:
        st.warning("Please enter a tag and name to save the chat.")
    
    if save and name:
        pickle.dump(msgs, open(f"{name}.pkl", "wb"))
        st.success(f"Chat saved")