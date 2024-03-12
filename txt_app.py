import streamlit as st
from io import StringIO
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import tiktoken
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css,bot_template, user_template
import tempfile
import os


def get_text_chunks(documents):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(documents)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm= ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain= ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    #this sets up memory to remember previous conversation
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2==0:
            st.write(user_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with Text Documents', page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None 
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    
    st.header("Chat with Text Documents :books:")
    user_question = st.text_input('Ask a question about your documents:')
    if user_question:
        handle_userinput(user_question)

    # st.write(user_template.replace('{{MSG}}','hello bot'), unsafe_allow_html=True)
    # st.write(bot_template.replace('{{MSG}}', 'hello human'), unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("Your documents")
        txt_docs = st.file_uploader("Upload your Text Documents here and click on 'Process'", type=['txt'], accept_multiple_files = False)
        if st.button("Process"):
            #to ensure that user knows the process button is running and not frozen
            with st.spinner('Processing'):
                if txt_docs is not None:
                    documents = txt_docs.read()
                    documents = str(documents)
                
                #get pdf text
                #returns single string of text from all pdfs
                
                
                #get text chunks
                text_chunks = get_text_chunks(documents)
                
                #create vector store
                vectorstore = get_vectorstore(text_chunks)

                #create conversation chain
                #allows new messages during conversation
                #makes sure streamlit does not reload the entire page when an action is provoked. it remembers conversation
                st.session_state.conversation = get_conversation_chain(vectorstore)
    




if __name__=='__main__':
    main()