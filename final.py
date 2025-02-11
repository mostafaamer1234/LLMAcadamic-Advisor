from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
import pandas as pd
import os

LANGCHAIN_PROJECT = "pr-notable-garbage-99"
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ[
    'OPENAI_API_KEY'] = "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"

#### INDEXING ####

# File paths
pdf_path = "/Users/mostafa/PycharmProjects/LLM/whatIfReport.pdf"
csv_path = "/Users/mostafa/PycharmProjects/LLM/processed_psu_courses.csv"

# Load PDF documents
loader = PyPDFLoader(pdf_path)
pdf_docs = loader.load()

# Create the CSV agent
csv_agent = create_csv_agent(
    llm=OpenAI(temperature=0),
    path=csv_path,
    allow_dangerous_code=True  # Opt-in to allow dangerous code execution
)

# Split PDF documents for RAG
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pdf_splits = text_splitter.split_documents(pdf_docs)

# Create a vectorstore for RAG pipeline
vectorstore = Chroma.from_documents(documents=pdf_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM for RAG
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain for RAG
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


#### ROUTING LOGIC ####

def select_data_source(question):
    """
    Route the query to the correct data source based on its content.
    """
    # Simplistic logic to decide based on keywords
    if "csv" in question.lower() or "course" in question.lower() or "prerequisite" in question.lower():
        return "csv"
    return "pdf"


def handle_question(question):
    """
    Process the question by selecting the correct data source and using the appropriate pipeline.
    """
    data_source = select_data_source(question)

    if data_source == "csv":
        print("Using CSV agent...")
        response = csv_agent.run(question)
    else:
        print("Using PDF RAG pipeline...")
        response = rag_chain.invoke(question)

    return response


# Example Questions
csv_question = "What are the prerequisites needed for Phys 212?"
pdf_question = "how many Humanities(GH) do I need?"

# Handle queries
print("CSV Query Result:")
csv_result = handle_question(csv_question)
print(csv_result)

print("\nPDF Query Result:")
pdf_result = handle_question(pdf_question)
print(pdf_result)
