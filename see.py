import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.agents import create_csv_agent
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List

# Environment Variables
LANGCHAIN_PROJECT = "pr-notable-garbage-99"
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'insert you key here'
os.environ['OPENAI_API_KEY'] = 'insert you key here'

#### FILE PATHS ####
pdf_path = "/Users/mostafa/PycharmProjects/LLM/whatIfReport.pdf"
courses_csv_path = "/Users/mostafa/PycharmProjects/LLM/processed_psu_courses.csv"
requirements_csv_path = "/Users/mostafa/PycharmProjects/LLM/major_requirements.csv"

#### INDEXING ####
# Load PDF documents
loader = PyPDFLoader(pdf_path)
pdf_docs = loader.load()

# Split PDF documents for RAG
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
pdf_splits = text_splitter.split_documents(pdf_docs)

# Create a vectorstore for RAG pipeline
vectorstore = Chroma.from_documents(documents=pdf_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Create CSV agents for courses and major requirements
courses_agent = create_csv_agent(
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
    path=courses_csv_path,
    allow_dangerous_code=True
)

requirements_agent = create_csv_agent(
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
    path=requirements_csv_path,
    allow_dangerous_code=True
)

#### QUERY TRANSLATION ####

def translate_query(question):
    translation_prompt = [
        SystemMessage(content="You are an expert query translator. Translate the user's question into a structured query for retrieval."),
        HumanMessage(content=f"Original Question: {question}")
    ]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    translated_query = llm.invoke(translation_prompt)
    return translated_query.content.strip()

#### QUERY ROUTING ####
class RouteQuery(BaseModel):
    datasources: List[str] = Field(...,
                                   description="List of data sources needed: ['processed_psu_courses.csv', 'major_requirements.csv', 'whatIfReport.pdf']")


llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery, method="function_calling")


def route_question(question):
    routing_prompt = [
        SystemMessage(content="Route the user's question based on its context. "
                              "If the question is about courses description/ definition, prerequisites, or specific course credits, route it to 'processed_psu_courses.csv'. "
                              "For major requirements or minor requirments for graduation, choose 'major_requirements.csv'. "
                              "For general academic requirements or unit credit or includes academic category codes like: GH, CMPAB_BS, GS, etc... , route it to choose 'pdf'."
                              ' If the question is about how many more classes left, or what classes should I take next, then route to  all sources, the two csv files and the pdf file'),
        HumanMessage(content=question)
    ]
    routing_result = structured_llm.invoke(routing_prompt)
    return routing_result.datasources


#### HANDLING USER QUERIES ####

def handle_csv_query(agent, question):
    return agent.run(question).strip()


def handle_pdf_query(question):
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content[:500] for doc in relevant_docs])
    pdf_llm_prompt = [
        SystemMessage(content="You are an academic assistant. Answer the question based on the context provided."
                              "If the question is about how many more classes left to graduate, or what classes should I take next, then first the use  major_requirements.csv to get data about all the courses required for the major/minor. then use the data in the whatIfReport.pdf (which has all the courses and units the student has completed + the units they still need) to see all the courses the student has already taken and all the units still required. Then use processed_psu_courses.csv to check for the specific courses that satisfy the major requirments and the unit requirments. Using all this data from all the sources, make a comprehensive accurate list of all the clases the student needs to graduate '"),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ]
    answer = llm.invoke(pdf_llm_prompt)
    return answer.content.strip()


def handle_question(question):
    translated_question = question  # Directly using question; modify if needed
    data_sources = route_question(translated_question)
    responses = []

    if "processed_psu_courses.csv" in data_sources:
        responses.append(handle_csv_query(courses_agent, translated_question))
    if "major_requirements.csv" in data_sources:
        responses.append(handle_csv_query(requirements_agent, translated_question))
    if "whatIfReport.pdf" in data_sources:
        responses.append(handle_pdf_query(translated_question))

    combined_response = "\n\n".join(responses)
    if len(combined_response) > 5000:
        combined_response = combined_response[:5000] + "... (truncated for brevity)"

    return combined_response

#### MAIN ####
if __name__ == "__main__":
    user_question = input("Ask your question: ")
    print("\nProcessing your query...\n")
    result = handle_question(user_question)
    print("\nResult:")
    print(result)
