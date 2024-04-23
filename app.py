import streamlit as st
from generate_response import generate_response
from fetch_content import storing_pinecone, fetch

st.title("PDF Knowledge-base Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []


if "data_embedded" not in st.session_state:
    st.session_state.data_embedded = False


upload_file = st.file_uploader("Upload PDFs for context", type=["PDF", "pdf"])


if upload_file:
    with st.spinner('Loading...'):
        file_name = upload_file.name
        st.session_state.data_embedded = True


if st.session_state.data_embedded:
    if prompt := st.chat_input("Ask me anything!"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner('Answering your query...'):
            response = generate_response(prompt)

        st.session_state.messages.append(
            {"role": "assistant", "content": response})


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
