import os
import pandas as pd
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field

# Environment Variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['OPENAI_API_KEY'] = "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"

# File paths
pdf_path = "/Users/mostafa/PycharmProjects/LLM/whatIfReport.pdf"
csv_path = "/Users/mostafa/PycharmProjects/LLM/psu_courses.csv"


def load_pdf(file_path):
    """Load content from a PDF file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_csv(file_path):
    """Load and preprocess CSV content."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path)
    df = df.dropna()  # Remove empty rows

    # Process rows into Document objects
    csv_docs = []
    for _, row in df.iterrows():
        csv_content = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
        csv_docs.append(Document(page_content=csv_content, metadata={"source": file_path}))
    return csv_docs


def format_docs(docs):
    """Format retrieved documents for input into the LLM."""
    return "\n\n".join(doc.page_content for doc in docs)


def initialize_vectorstore(pdf_docs, csv_docs):
    """Create a vectorstore retriever from documents."""
    all_docs = pdf_docs + csv_docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever(search_kwargs={"k": 3})  # Limit to top 3 results


def query_system(question, retriever, llm):
    """Query the system and return formatted results."""
    relevant_docs = retriever.invoke(question)
    if not relevant_docs:
        return "No relevant information found."

    formatted_context = format_docs(relevant_docs)

    # Query the LLM
    llm_inputs = [
        SystemMessage(content="Based on the provided context, answer the question."),
        HumanMessage(content=f"Context:\n{formatted_context}\n\nQuestion: {question}")
    ]
    response = llm.invoke(llm_inputs)
    answer = response.content.strip()  # Extract the content from the AIMessage object
    sources = {doc.metadata["source"] for doc in relevant_docs}
    return f"{answer} retrieved from {', '.join(sources)}"


# Main Script
if __name__ == "__main__":
    try:
        # Load data
        pdf_docs = load_pdf(pdf_path)
        csv_docs = load_csv(csv_path)

        # Initialize retriever and LLM
        retriever = initialize_vectorstore(pdf_docs, csv_docs)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Example query
        question = "What is math 141?"
        result = query_system(question, retriever, llm)
        print(result)
    except Exception as e:
        print(f"Error: {e}")