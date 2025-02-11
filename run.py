from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import HumanMessage, SystemMessage
import os

# Environment Variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ[
    'OPENAI_API_KEY'] = "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"

# Load and split documents
datasource = ["/Users/mostafa/PycharmProjects/LLM/whatIfReport.pdf",
              "/Users/mostafa/PycharmProjects/LLM/psu_courses.pdf"]

docs = []
for source in datasource:
    loader = PyPDFLoader(source)
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create a vectorstore and retriever
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Define LLM with function output using the `function_calling` method
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


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
    return "\n\n".join(chunk.page_content for chunk in chunks)


# Define the pipeline
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
question = "What what Humanities(GH) course credits do I still need??"

result = get_answer(question)
print(result)
