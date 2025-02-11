from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import os
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
import pandas as pd
import pypdf
from typing import Literal
#from langchain_core.pydantic_v1 import BaseModel, Field

LANGCHAIN_PROJECT="pr-notable-garbage-99"
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'insert you key here'
os.environ['OPENAI_API_KEY'] = 'insert you key here'

#### INDEXING ####


file_path = "/Users/mostafa/PycharmProjects/LLM/whatIfReport.pdf"
#file_path2 = "/Users/mostafa/PycharmProjects/LLM/psu_courses.pdf"# Replace with your document's path
loader = PyPDFLoader(file_path)
docs = loader.load()

df = pd.read_csv('/Users/mostafa/PycharmProjects/LLM/processed_psu_courses.csv')
csv_path = "/Users/mostafa/PycharmProjects/LLM/psu_courses.csv"

# Create the CSV agent
agent = create_csv_agent(
    llm=OpenAI(temperature=0),
    path=csv_path,
    allow_dangerous_code=True  # Opt-in to allow dangerous code execution
)

# Example query
question = "What are the prerequisites needed for Phys 212?"
response = agent.run(question)
print(response)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
result = rag_chain.invoke("what Humanities(GH) course did i take?")
print(result)
