import streamlit as st
from retriever import medical_query_rag
from langgraph.graph import MessagesState
import time

st.set_page_config(page_title="Healthcare Chatbot", layout="wide")
st.title("Healthcare Chatbot 🤖🩺")
st.write("Ask any healthcare question. Responses are generated using trusted sources.")


if "state" not in st.session_state:
    st.session_state.state = MessagesState(messages=[])
if "messages" not in st.session_state:
    st.session_state.messages = []

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
