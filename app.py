import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import openai
import os
import shutil
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Amotions Demo",
                   page_icon="https://www.amotionsinc.com/navbar-logo.svg")
st.image("https://www.amotionsinc.com/navbar-logo.svg")
st.title("Amotions demo")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
        st.session_state.conversation = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = None


def get_vectorstore(uploaded_files):
    # text = ""
    # for pdf in pdf_docs:
    #     pdf_reader = PdfReader(pdf)
    #     for page in pdf_reader.pages:
    #         text += page.extract_text()
    # return text
    if os.path.exists("data"):
        shutil.rmtree("data")
    os.makedirs("data")
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        filename = os.path.join("data", uploaded_file.name)  # Path to save the file in the "data" directory

         # Save the file in the "data" directory
        with open(filename, "wb") as f:
            f.write(bytes_data)

        st.sidebar.write(f"File '{uploaded_file.name}' saved in the 'data' directory.")
    loader = PyPDFDirectoryLoader('data')
    all_docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.split_documents(all_docs)
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

    return vectorstore



def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    return response["answer"]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def main():
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # create vector store
                vectorstore = get_vectorstore(pdf_docs)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

    if st.session_state.conversation:
        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                # full_response = ""
                full_response = handle_userinput(prompt)
                # for response in openai.ChatCompletion.create(
                #     model=st.session_state["openai_model"],
                #     messages=[
                #         {"role": m["role"], "content": m["content"]}
                #         for m in st.session_state.messages
                #     ],
                #     stream=True,
                # ):
                #     full_response += response.choices[0].delta.get("content", "")
                #     message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
    else:
        st.info("Upload documents")

if __name__ == '__main__':
    main()