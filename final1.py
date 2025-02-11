from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.agents import create_csv_agent
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Literal
import os

# Environment Variables
LANGCHAIN_PROJECT = "pr-notable-garbage-99"
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'insert you key here'
os.environ['OPENAI_API_KEY'] = 'insert you key here'

#### INDEXING ####

# File paths
pdf_path = "/Users/mostafa/PycharmProjects/LLM/whatIfReport.pdf"
csv_path = "/Users/mostafa/PycharmProjects/LLM/processed_psu_courses.csv"

# Load PDF documents
loader = PyPDFLoader(pdf_path)
pdf_docs = loader.load()

# Create the CSV agent
csv_agent = create_csv_agent(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    path=csv_path,
    allow_dangerous_code=True  # Opt-in to allow dangerous code execution
)

# Split PDF documents for RAG
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pdf_splits = text_splitter.split_documents(pdf_docs)

# Create a vectorstore for RAG pipeline
vectorstore = Chroma.from_documents(documents=pdf_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

#### ROUTING LOGIC ####

# Define the routing model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["csv", "pdf"] = Field(
        ...,
        description="Choose 'csv' for course-related queries or 'pdf' for general academic queries."
    )

# LLM with function call for routing
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery, method="function_calling")

# Routing prompt template
def create_routing_prompt(question):
    """
    Creates the routing prompt to decide the data source.
    """
    system_message = SystemMessage(
        content="You are an expert at routing a user question to the appropriate data source. "
                "Given a user question, decide if it refers to courses or prerequisites (CSV source) "
                "or academic requirements (PDF source)."
    )
    human_message = HumanMessage(content=question)
    return [system_message, human_message]


def route_question(question):
    """
    Uses the LLM to decide whether the question refers to the CSV or PDF data source.
    """
    routing_prompt = create_routing_prompt(question)
    routing_result = structured_llm.invoke(routing_prompt)
    return routing_result.datasource


#### HANDLING USER QUERY ####

def handle_question(question):
    """
    Process the question by selecting the correct data source and using the appropriate pipeline.
    """
    data_source = route_question(question)

    if data_source == "csv":
        print("Using CSV agent...")
        response = csv_agent.run(question)
    else:
        print("Using PDF RAG pipeline...")
        formatted_context = "\n\n".join([doc.page_content for doc in retriever.invoke(question)])
        response = f"Context: {formatted_context}\nQuestion: {question}"

    return response


#### MAIN ####

if __name__ == "__main__":
    user_question = input("Ask your question: ")
    print("\nProcessing your query...\n")
    result = handle_question(user_question)
    print("\nResult:")
    print(result)

