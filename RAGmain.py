import os
import constant as constant
os.environ['ANTHROPIC_API_KEY'] = constant.anthropic_key
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = constant.smith_key
os.environ["COHERE_API_KEY"] = constant.cohere_key

import bs4
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st


st.title("Guvi FAQ chatbot")
st.divider()


## RAG
@st.cache_data
def load_data():
    loader = BSHTMLLoader("guvi-faq.html")
    data = loader.load()
    return data

#print("Type of loaded data:", type(data))
#if isinstance(data, list) and len(data) > 0:
    # print("Type of first document in data:", type(data[0]))
    #content = data[0].page_content


@st.cache_resource
def create_vectorstore(_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    all_splits = text_splitter.split_documents(data) # indexing: Split
    embed = CohereEmbeddings(model="embed-english-light-v3.0")  # indexing: Store
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed)
    return vectorstore


data = load_data()
vectorstore = create_vectorstore(data)
retriever = vectorstore.as_retriever(search_type="similarity") # Reteive


llm = ChatAnthropic(model="claude-3-sonnet-20240229",temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a chatbot who will answer FAQ for Guvi."),
    ("user","{question}")
])
output_parser = StrOutputParser()
def format_docs(data):
    return "\n\n".join(doc.page_content for doc in data)

# chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)


#st.title("Guvi Chat bot")
#question = str(input("Enter your query"))
#result = rag_chain.invoke("who founded guvi")
#print(result)

#for chunk in rag_chain.stream("what is guvi"):
    #print(chunk, end="", flush=True)


## streamlit
#initialise a store space
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# take input; store it with role:user and display it
if question := st.chat_input("Enter your query"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

# generate output; store it with role:assistant and display it
with st.chat_message("assistant"):
    result = rag_chain.invoke(question)
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.markdown(result)


