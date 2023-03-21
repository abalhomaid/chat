import streamlit as st
import openai
from datetime import datetime
from streamlit.components.v1 import html
import pandas as pd
import csv
import os

from langchain.document_loaders import UnstructuredFileLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

os.environ['HUGGINGFACEHUB_API_TOKEN'] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
loader = UnstructuredFileLoader("files/Tasrif.pdf", strategy="fast", mode="elements")
docs = loader.load()

chain = load_qa_chain(HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64}), 
                      chain_type="stuff")

st.set_page_config(page_title="Ask about Tasrif")




html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """



with st.sidebar:
    st.markdown("""
    # About 
    Ask a question about Tasrif paper.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    # How does it work
    Runs on google/flan-t5-xl model. The language model to generate text based not on generic data but rather on specific data (Tasrif).
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""

    Inspired by [@nainia_ayoub](https://twitter.com/nainia_ayoub) Brainstorm streamlit app
    """,
    unsafe_allow_html=True,
    )


input_text = None
if 'output' not in st.session_state:
    st.session_state['output'] = 0

if st.session_state['output'] <=2:
    st.markdown("""
    # Ask questions on Tasrif
    """)
    input_text = st.text_input("", disabled=False, placeholder="Type question")
    st.session_state['output'] = st.session_state['output'] + 1
else:
    st.info("Thank you! Refresh for more questions💡")
    st.markdown('''
    <a target="_blank" style="color: black" href="https://twitter.com/intent/tweet?text=Checkout%20Tasrif%20paper%20streamlit%20question%20answering%20app%20by%20@abalhomaid%0A%0Ahttps://abalhomaid.github.io/">
        <button class="btn">
            Tweet about this!
        </button>
    </a>
    <style>
    .btn{
        display: inline-flex;
        -moz-box-align: center;
        align-items: center;
        -moz-box-pack: center;
        justify-content: center;
        font-weight: 400;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        margin: 0px;
        line-height: 1.6;
        color: #fff;
        background-color: #00acee;
        width: auto;
        user-select: none;
        border: 1px solid #00acee;
        }
    .btn:hover{
        color: #00acee;
        background-color: #fff;
    }
    </style>
    ''',
    unsafe_allow_html=True
    )

hide="""
<style>
footer{
	visibility: hidden;
    position: relative;
}
.viewerBadge_container__1QSob{
    visibility: hidden;
}
#MainMenu{
	visibility: hidden;
}
<style>
"""
st.markdown(hide, unsafe_allow_html=True)

# html(button, height=70, width=220)
st.markdown(
    """
    <style>
        iframe[width="220"] {
            position: fixed;
            bottom: 60px;
            right: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
if input_text:
    prompt = "" + str(input_text)
    if prompt:
        response = chain.run(input_documents=docs[:5], question=prompt)

        tasrif_output = response
        today = datetime.today().strftime('%Y-%m-%d')
        topic = "Thinking about a response: "+input_text+"\n@Date: "+str(today)+"\n"+tasrif_output
        
        st.info(tasrif_output)
        filename = "tasrif_"+str(today)+".txt"
        btn = st.download_button(
            label="Download txt",
            data=topic,
            file_name=filename
        )
        fields = [input_text, tasrif_output, str(today)]
        # read local csv file
        # r = pd.read_csv('./data/prompts.csv')
        if len(fields)!=0:
            with open('./data/prompts.csv', 'a', encoding='utf-8', newline='') as f:
                # write to csv file (append mode)
                writer = csv.writer(f, delimiter=',', lineterminator='\n')
                writer.writerow(fields)