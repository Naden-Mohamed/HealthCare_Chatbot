import streamlit as st
from retriever import medical_query_rag
from langgraph.graph import MessagesState
from doc_loader import CustomDocumentLoader
import time

st.set_page_config(page_title="Healthcare Chatbot", layout="wide")
st.title("Healthcare Chatbot 🤖🩺")
st.write("Ask any healthcare question. Responses are generated using trusted sources.")



if "state" not in st.session_state:
    st.session_state.state = MessagesState(messages=[])
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_content" not in st.session_state:
    st.session_state.file_content = None
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None



def ask_your_file(query):
    if not st.session_state.file_content:
        return "No file content available. Please upload a document first."

    answer = CustomDocumentLoader.preprocess_file_content(query, st.session_state.file_content, state=st.session_state.state)
    return answer


# Sidebar for file upload to answer questions based on file content
with st.sidebar:
    st.session_state.uploaded_file = st.sidebar.file_uploader(
        "Upload a document", type=["pdf", "txt", "docx", "pptx"]
    )
    if st.session_state.uploaded_file:
        st.session_state.file_path = CustomDocumentLoader.save_uploaded_file(st.session_state.uploaded_file)
        loader = CustomDocumentLoader(st.session_state.file_path)
        st.session_state.file_content = [doc.page_content for doc in loader.lazy_load()]
        st.success(f"{st.session_state.uploaded_file.name} loaded successfully!")


    if st.sidebar.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.file_content = None

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



user_query = st.chat_input("Ask me anything about healthcare!") 

if user_query:

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)


    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_text = ""

        if st.session_state.uploaded_file:
            answer_obj = ask_your_file(user_query)
        else:
            answer_obj = medical_query_rag(user_query, state=st.session_state.state)


        # Correctly extract latest AI response from state
        if isinstance(answer_obj, dict) and "messages" in answer_obj:
            last_message = answer_obj["messages"]
            answer_text = getattr(last_message, "content", str(last_message))
        else:
            answer_text = str(answer_obj)

        for token in answer_text.split():
            response_text += token + " "
            response_placeholder.markdown(response_text)
            time.sleep(0.05)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
