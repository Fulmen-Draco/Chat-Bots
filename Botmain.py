import os
from constant import anthropic_key
os.environ["ANTHROPIC_API_KEY"] = anthropic_key

from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

st.title("Programming Chat bot")
st.divider()
 
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.2, max_tokens_to_sample=512)
output_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a proficient {Language} Programmer, and will now assist me with by solving my doubts."),
    ("user","{query}")
])

chain = prompt | llm | output_parser

# defining inputs
with st.sidebar:
    p_lang = st.text_input("Enter name of programming language",key="lang")



#if doubt:
 #   st.write(chain.invoke({"Language":p_lang,"query":doubt}))

## streamlit
#initialise a store space
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# take input; store it with role:user and display it
if doubt := st.chat_input("Enter your doubt",key="doubt"):
    st.session_state.messages.append({"role": "user", "content": doubt})
    with st.chat_message("user"):
        st.markdown(doubt)

# generate output; store it with role:assistant and display it
    with st.chat_message("assistant"):
        result = chain.invoke({"Language":p_lang,"query":doubt})
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.markdown(result)



