from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import os
import pypdf

import os



os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['OPENAI_API_KEY'] = "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"


# Two prompts
datasource = ["/Users/mostafa/PycharmProjects/LLM/whatIfReport.pdf", "/Users/mostafa/PycharmProjects/LLM/psu_courses.pdf"]
loader = PyPDFLoader(datasource)
docs = loader.load()


class RouteQuery(BaseModel):
    """Based on the datasource most relevant to the user's question, find the answer to the user's question."""

    datasource: Literal[
        "/Users/mostafa/PycharmProjects/LLM/whatIfReport.pdf", "/Users/mostafa/PycharmProjects/LLM/psu_courses.pdf"] = Field(
        ...,
        description="Answer the user's question from the datasource most relevant to the question",
    )


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

# Prompt
system = """Based on the the datasource most relevant to the user's question, find the answer to the user's question."""
question = "what GA classes have I taken so far?"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Define router
router = prompt | structured_llm


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in datasource)



rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | router
    | StrOutputParser()
)

result = rag_chain.invoke({"question": question})
print(result)