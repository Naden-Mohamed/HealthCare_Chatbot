import streamlit as st
from retriever import medical_query_rag  # remove .py

st.set_page_config(page_title="Healthcare Chatbot", layout="wide")
st.title("Healthcare Chatbot 🤖🩺")

st.write("This chatbot provides information on healthcare topics using trusted sources like Wikipedia and medical websites.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)

# User input
user_query = st.chat_input("Ask me anything about healthcare!")
if user_query:

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Fetch AI response
    with st.spinner("Fetching answer..."):
        response = medical_query_rag(user_query, top_k=500, top_n_chuncks=500)

    # Append bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
