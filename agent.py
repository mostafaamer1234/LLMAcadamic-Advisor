import os
import pandas as pd
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI

# Environment Variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'insert you key here'
os.environ['OPENAI_API_KEY'] = 'insert you key here'



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