import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.markdown(
    """
    ## Generate Powerful Questions & Conquer Your Studies!

    Welcome to Brainiac Buddy's question generation zone! Here, you can transform your PDFs into personalized learning experiences. 

    """
)
topic = st.text_input("Enter the topic you want to practice")

num_questions = st.slider("Select the number of questions to generate", 5, 10)

uploaded_files = st.file_uploader("Upload study material", accept_multiple_files=True, type=["pdf"])

generate_btn = st.button("Generate Questions", type="primary")
if not topic:
    st.warning("Please enter a topic to generate questions.")

# _______________________________________________________________________ #
llm = ChatNVIDIA(model="meta/llama3-70b-instruct", nvidia_api_key="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", seed=42)
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

# Prompts
system_prompt = (
    "You are an assistant to generate questions and answers. "
    "Use the following pieces of retrieved context to generate questions and their answers. "
    "If the given context is not enough to generate a question, "
    "you should reply that the given context is not sufficient. "
    "Example: "
    "Questions: "
    "1. What is the capital of France? "
    "2. What is the capital of Spain? "
    "3. What is the capital of Italy? "
    "Answers: "
    "1. Paris "
    "2. Madrid "
    "3. Rome "
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retrieval_chain = create_retrieval_chain(retriever, docs_chain)

def gen_questions(num_questions: int, topic: str):
    gen_prompt = f'''
    Please generate {num_questions} questions using the context from the documents on the topic "{topic}".
    Generate long-form/medium-form questions and answers based on the topic and the context.
    Use the following structure:
    ```
    Questions:
    1. question 1
    2. question 2
    ...
    Answers:
    1. answer 1<br><br>
    2. answer 2<br><br>
    ...
    ```
    '''
    return retrieval_chain.invoke({"input": gen_prompt})

# _______________________________________________________________________ #

if generate_btn and topic and uploaded_files:
    with st.spinner(f"Generating {num_questions} questions ..."):
        res = gen_questions(num_questions, topic)
        # Split questions and answers
        question_answers = res['answer'].split('Answers:')
        # Show questions
        st.markdown(question_answers[0])
        # Show answers when toggled
        try:
            st.markdown(f"<details><summary>Show Answers</summary>{question_answers[1]}</details>", unsafe_allow_html=True)
        except IndexError:
            st.warning("The documents provided is not sufficient to generate questions. Please provide more context or try changing the input documents.")