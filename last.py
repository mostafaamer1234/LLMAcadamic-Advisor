import os
import pandas as pd
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import HumanMessage, SystemMessage, Document

# Environment Variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'insert you key here'
os.environ['OPENAI_API_KEY'] = 'insert you key here'

# File paths
pdf_source = "/Users/mostafa/PycharmProjects/LLM/whatIfReport.pdf"
csv_source = "/Users/mostafa/PycharmProjects/LLM/processed_psu_courses.csv"

# Load and process PDF document
loader = PyPDFLoader(pdf_source)
docs = loader.load()

# Load and process CSV file
if os.path.exists(csv_source):
    df = pd.read_csv(csv_source)
    # Limit rows for efficiency
    csv_content = df.head(100).to_string(index=False)
    csv_doc = Document(page_content=csv_content, metadata={"source": csv_source})
    docs.append(csv_doc)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# Create a vectorstore and retriever
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Limit to top 5 results

# Define LLM with function output using the `function_calling` method
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


class RouteQuery(BaseModel):
    """Output schema for determining the source document and answering the query."""
    datasource: str = Field(..., description="Path of the most relevant datasource.")
    answer: str = Field(..., description="Answer based on the relevant datasource.")


structured_llm = llm.with_structured_output(RouteQuery, method="function_calling")

# Prompt messages
system_message = SystemMessage(
    content="Based on the datasource most relevant to the user's question, find the answer to the user's question.")


# Function to convert document chunks to a formatted string
def format_docs(chunks):
    # Limit to top 3 chunks for efficiency
    top_chunks = chunks[:3]
    return "\n\n".join(chunk.page_content for chunk in top_chunks)


# Define the pipeline
def get_answer(question):
    # Retrieve relevant documents
    relevant_docs = retriever.invoke(question)

    # Format retrieved documents for the LLM
    formatted_context = format_docs(relevant_docs)

    # Prepare inputs as a list of messages
    llm_inputs = [
        system_message,
        HumanMessage(content=f"Context: {formatted_context}\n\nQuestion: {question}")
    ]

    # Generate the structured output using `function_calling`
    structured_output = structured_llm.invoke(llm_inputs)

    # Format the output as desired
    formatted_output = f"{structured_output.answer} retrieved from {structured_output.datasource}"
    return formatted_output


# Query
question = "What are the prerequisites needed for Phys 212"

result = get_answer(question)
print(result)

