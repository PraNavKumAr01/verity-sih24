import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.tools import Tool
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os
import time
import warnings
import json
import re

warnings.filterwarnings("ignore")
load_dotenv()

os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["TOKENIZERS_PARALLELISM"] = st.secrets["TOKENIZERS_PARALLELISM"]
os.environ["GOOGLE_CSE_ID"] = st.secrets["GOOGLE_CSE_ID"]
os.environ["GOOGLE_API_KEY"] =  st.secrets["GOOGLE_API_KEY"]
vectorstore_index_name = st.secrets["VECTORSTORE_INDEX_NAME"]

@st.cache_resource
def initialize_qa_chain():
    llm = ChatGroq(temperature=0.3, model_name="llama3-70b-8192")

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    
    pc = Pinecone(api_key= st.secrets["PINECONE_API_KEY"])  
    spec = ServerlessSpec(cloud='aws', region='us-east-1')  
    if vectorstore_index_name in pc.list_indexes().names():  
        pc.delete_index(vectorstore_index_name)  
    pc.create_index(  
        vectorstore_index_name,  
        dimension=768,
        metric='dotproduct',  
        spec=spec  
    )

    vectorstore = PineconeVectorStore(
            index_name = vectorstore_index_name,
            embedding = embeddings,
            pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        )

    prompt = hub.pull("rlm/rag-prompt-llama")
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

@st.cache_resource
def initialize_google_search():
    search = GoogleSearchAPIWrapper()
    def get_links(query):
        return search.results(query + "indian laws", 5)

    def search_func(query):
        return search.run(query + "indian laws")

    links_tool = Tool(
        name="Google Search Links",
        description="Search Google for recent snippets and links",
        func=get_links,
    )

    search_tool = Tool(
        name="Google Search Result",
        description="Search Google for recent results.",
        func=search_func,
    )
    
    return links_tool, search_tool

@st.cache_resource
def get_llm():
    return ChatGroq(temperature=0.3, model_name="llama3-70b-8192")

def get_web_search_answer(llm, query, search_result):
    prompt = f"""
    Based on the following search result, please provide a concise and informative answer to the question: "{query}"

    Search Result:
    {search_result}

    Please synthesize the information and provide a clear, coherent answer. If there are any limitations or uncertainties in the information, please mention them.
    """
    return llm.invoke(prompt)

def extract_query_info(llm, query):
    prompt = f"""
    Analyze the following query and extract the following information:
    1. The likely state of legislation the user is referring to
    2. Any keywords related to legal concepts (e.g., contract, automobile)
    3. Names of any acts mentioned

    Query: {query}

    Please provide your response in JSON format with the following structure:
    {{
        "state": "name of state or 'Unknown' if not clear",
        "keywords": ["list", "of", "keywords"],
        "acts": ["list", "of", "act", "names"]
    }}

    If you cant deduce the state from the given query, set the default state to 'India'

    Your response should contain only the json and no other information before or after the json.
    """
    response = llm.invoke(prompt)
    return json.loads(response.content)

def init_browser():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def get_search_results(search_query,list):
    driver = init_browser()

    try:
        driver.get("https://www.aironline.in/")
        time.sleep(2) 
        search_bar = driver.find_element(By.ID, "searchDatabaseTabOne")
        search_bar.clear()
        search_bar.send_keys(search_query)
        search_bar.send_keys(Keys.RETURN)

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "newIdsearchResult_body")))

        for i in list:
            search_bar = driver.find_element(By.ID, "searchBox")
            search_bar.clear()
            search_bar.send_keys(i)
            search_bar.send_keys(Keys.RETURN)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "newIdsearchResult_body")))

        search_button = driver.find_element(By.ID, "judgmentTabId")
        search_button.click()

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "newIdsearchResult_body")))
        buttons = driver.find_elements(By.CLASS_NAME, 'btn azure showInstancesDiv') 

        for button in buttons:
            button.click()
            time.sleep(2)

        buttons2 = driver.find_elements(By.CLASS_NAME, 'viewFilterAnchor')
        for bt in buttons2:
            driver.execute_script("arguments[0].click();", bt)
            time.sleep(2)

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        results = soup.find_all(id="rightArticle_Id")

        search_results = []
        for result in results:
            search_results.append(result.get_text(strip=True))

        split_text = re.split(r'\d+\.Judgment', search_results[0])
        split_text = [s.strip() for s in split_text if s.strip()]

        return split_text

    finally:
        driver.quit()

def extract_judge(text):

    split_text2 = re.split(r'Full JudgmentView', text)
    split_text2 = [s.strip() for s in split_text2 if s.strip()]
    split_text3 = re.split(r'Judge',split_text2[0] )
    split_text3 = [s.strip() for s in split_text3 if s.strip()]
    cleaned_string = re.sub(r'\s+', ' ', split_text3[1]).strip()
    cleaned_string = cleaned_string[4:]
    l=cleaned_string.split(",")

    return l

def extract_headnote(text):
    head_present = bool(re.search(r'Acts:', text))
    if head_present==False:
      return
    split = re.split(r'Headnote:', text)
    split = [s.strip() for s in split if s.strip()]
    acts_present = bool(re.search(r'Acts:', text))
     
    if acts_present==False:
      return split[1]
    else:
      split1= re.split(r'Acts:', split[1])
      split1=[s.strip() for s in split1 if s.strip()]
      return split1[0]

def extract_judgment(text):
    judment_present = bool(re.search(r'Judgment Text:', text))
    if judment_present==False:
        return

    split = re.split(r'Judgment Text:', text)
    split = [s.strip() for s in split if s.strip()]
    split1= re.split(r'Headnote:', split[1])
    split1=[s.strip() for s in split1 if s.strip()]
    
    return split1[0]

def create_case_template():
    return {
        "judges": [],
        "headnote": "",
        "judgement": ""
    }

def process_text_and_fill_template(text_list):
    result = {
        "cases": []
    }

    for text in text_list:
        case = create_case_template()

        case["judges"] = extract_judge(text)
        case["headnote"] = extract_headnote(text)
        case["judgement"] = extract_judgment(text)
        result["cases"].append(case)

    return result

st.title("Legal Research Assistant")

qa_chain = initialize_qa_chain()
links_tool, search_tool = initialize_google_search()
llm = get_llm()

question = st.text_input("Enter your question:")
perform_case_analysis = st.checkbox("Perform Case Analysis")

if st.button("Get Answers"):
    if question:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RAG Answer")
            with st.spinner("Generating RAG answer..."):
                rag_result = qa_chain({"query": question})
                st.write(rag_result["result"])
        
        with col2:
            st.subheader("Web Search Answer")
            with st.spinner("Performing web search and generating answer..."):
                search_result = search_tool.run(question)
                web_answer = get_web_search_answer(llm, question, search_result)
                st.write(web_answer.content)
                
                st.subheader("Sources:")
                link_results = links_tool.run(question)
                for result in link_results:
                    st.write(f"- [{result['title']}]({result['link']})")
        
        if perform_case_analysis:
            st.subheader("Case Analysis")
            with st.spinner("Performing case analysis..."):
                query_info = extract_query_info(llm, question)
                st.write("Query Analysis:")
                st.json(query_info)
                
                search_query = f"{query_info['state']} {' '.join(query_info['keywords'])} {' '.join(query_info['acts'])}"
                split_text = get_search_results(search_query=search_query, list=query_info['keywords'])
                filled_template = process_text_and_fill_template(split_text)
                
                for i, case in enumerate(filled_template["cases"], 1):
                    st.subheader(f"Case {i}")
                    st.write("Judges:", ", ".join(case["judges"]))
                    st.write("Headnote:", case["headnote"])
                    with st.expander("View Full Judgment"):
                        st.write(case["judgement"])
    else:
        st.warning("Please enter a question.")
