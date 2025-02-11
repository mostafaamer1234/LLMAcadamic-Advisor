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
os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['OPENAI_API_KEY'] = "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"

#### INDEXING ####

# File paths
pdf_path = "/Users/mostafa/PycharmProjects/LLM/whatIfReport.pdf"
csv_path = "/Users/mostafa/PycharmProjects/LLM/processed_psu_courses.csv"

# Load PDF documents
loader = PyPDFLoader(pdf_path)
pdf_docs = loader.load()

# Create the CSV agent
csv_agent = create_csv_agent(
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
    path=csv_path,
    allow_dangerous_code=True  # Opt-in to allow dangerous code execution
)

# Split PDF documents for RAG
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pdf_splits = text_splitter.split_documents(pdf_docs)

# Create a vectorstore for RAG pipeline
vectorstore = Chroma.from_documents(documents=pdf_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Limit to top 3 chunks

#### QUERY TRANSLATION ####

def translate_query(question):
    """
    Translate the user's high-level question into a more structured and specific query.
    """
    translation_prompt = [
        SystemMessage(content="You are an expert query translator. Rewrite the user's question to be more specific "
                              "and structured for better retrieval. Provide only the translated query."),
        HumanMessage(content=f"Original Question: {question}")
    ]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    translated_query = llm.invoke(translation_prompt)
    return translated_query.content.strip()

#### ROUTING LOGIC ####

# Define the routing model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["csv", "pdf"] = Field(
        ...,
        description="Choose 'csv' for course-related queries or 'pdf' for general academic queries."
    )

# Define LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# LLM with function call for routing
structured_llm = llm.with_structured_output(RouteQuery, method="function_calling")

def route_question(question):
    """
    Uses the LLM to decide whether the question refers to the CSV or PDF data source.
    """
    routing_prompt = [
        SystemMessage(content="You are an expert at routing a user question to the appropriate data source. "
                              "If the question is about courses description/ definition, prerequisites, or specific course credits, route it to 'csv'. "
                              "For general academic requirements or unit credit or includes academic category codes like: GH, CMPAB_BS, GS, etc... , route it to 'pdf'."),
        HumanMessage(content=question)
    ]
    routing_result = structured_llm.invoke(routing_prompt)
    print(f"Routing decision: {routing_result.datasource}")  # Debugging output
    return routing_result.datasource

#### HANDLING USER QUERY ####

def handle_pdf_query(question):
    """
    Handles queries routed to the PDF data source.
    """
    relevant_docs = retriever.invoke(question)  # Retrieve the top 3 relevant chunks
    context = "\n\n".join([doc.page_content for doc in relevant_docs])  # Combine context

    # Use the LLM to generate a concise answer from the context
    pdf_llm_prompt = [
        SystemMessage(content="You are an academic assistant. Answer the question based on the context provided."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ]
    answer = llm.invoke(pdf_llm_prompt)
    return answer.content.strip()

def handle_question(question):
    """
    Process the question by translating it, routing it to the correct data source, and using the appropriate pipeline.
    """
    translated_question = translate_query(question)
    print(f"Translated Question: {translated_question}")  # Debugging output

    data_source = route_question(translated_question)

    if data_source == "csv":
        print("Using CSV agent...")
        response = csv_agent.run(translated_question)
    elif data_source == "pdf":
        print("Using PDF RAG pipeline...")
        response = handle_pdf_query(translated_question)
    else:
        print("Unrecognized data source. Defaulting to PDF.")
        response = handle_pdf_query(translated_question)

    return response

#### MAIN ####

if __name__ == "__main__":
    user_question = input("Ask your question: ")
    print("\nProcessing your query...\n")
    result = handle_question(user_question)
    print("\nResult:")
    print(result)
