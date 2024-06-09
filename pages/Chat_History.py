import os
import pickle
import streamlit as st

# Get all the .pkl files in the directory
def get_all_files():
    files = os.listdir()
    return [f for f in files if f.endswith('.pkl')]

# Get the name of the file without .pkl extension
def get_name(file):
    return file.split(".pkl")[0]

st.title("Chat History")
st.subheader("Select a chat to view")

history = st.selectbox("Select a chat", get_all_files(), format_func=get_name)

if history:
    chat = pickle.load(open(history, "rb"))
    avatars = {"human": "user", "ai": "assistant"}
    for msg in chat.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)