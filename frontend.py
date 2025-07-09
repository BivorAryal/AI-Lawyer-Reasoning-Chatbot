# Step 1: Setup Upload PDF functionality
import streamlit as st
from RAG_pipeline import answer_query, retrieve_docs, model, questions

#Load pdf file
upload_file = st.file_uploader("upload pdf",
                               type = "pdf",
                               accept_multiple_files = False)

# Step 2: Chatbot Skeleton (Question and Answers)
user_query = st.text_area("Enter your prompt: ", placeholder = "Ask Anything!")
ask_question = st.button("ASK AI Lawyer")

if ask_question:
    if upload_file:
        st.chat_message("user").write(user_query)
        
    # RAG Pipeline
        retrieved_docs = retrieve_docs(user_query)
        response = answer_query(retrieved_docs, model, user_query)
        #  dummy_response = "This is a Dummy Response."
        st.chat_message("AI Lawyer").write(response)
    else:
        st.error("Upload valid pdf!")