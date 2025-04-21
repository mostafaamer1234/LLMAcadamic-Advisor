import json
import os
import re

import pandas as pd
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from graphExperiment import course_graph, get_course_prerequisites

#  Environment Variables
LANGCHAIN_PROJECT = "<Your Langchain project name>"
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '<Your Langchain API KEY>'
os.environ['OPENAI_API_KEY'] = "<ypur OPEN-AI API KEY>"

#  File Paths
pdf_path = "/Users/mostafa/PycharmProjects/LLM/SAA_STD_DS (3).pdf"
courses_csv_path = "/Users/mostafa/PycharmProjects/LLM/all_abington_majors_combined(5).csv"


#  LLM Instances
llm1 = ChatOpenAI(model_name="gpt-4o", temperature=0)
llm2 = ChatOpenAI(model_name="gpt-4o", temperature=0)

import re

def redact_student_info(docs):
    redacted_docs = []

    for doc in docs:
        content = doc.page_content

        # Mask lines like "Name: John Doe" or "Mostafa Amer"
        content = re.sub(r"\b[A-Z][a-z]+\s[A-Z][a-z]+(?=:\s?\d{6,})", "[REDACTED NAME]", content)

        # Mask student ID formats (9-digit, common for PSU)
        content = re.sub(r"\b\d{9}\b", "[REDACTED ID]", content)

        # Optionally mask name appearing again (first + last)
        # This finds proper names appearing elsewhere like "John Doe"
        content = re.sub(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", "[REDACTED NAME]", content, count=1)

        doc.page_content = content
        redacted_docs.append(doc)

    return redacted_docs

def extract_completed_and_ip_courses(text):
    pattern = re.findall(r"\b[A-Z]{2,}\s+[0-9]{3}\b", text)
    courses = {re.sub(r"\s+", " ", course.strip()) for course in pattern}
    return courses

#  PDF (WhatIfReport) Processing
loader = PyPDFLoader(pdf_path)
pdf_docs = redact_student_info(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=130)
pdf_splits = text_splitter.split_documents(pdf_docs)
vectorstore_pdf = Chroma.from_documents(pdf_splits, embedding=OpenAIEmbeddings())
pdf_retriever = vectorstore_pdf.as_retriever(search_kwargs={"k": 37})

#  Process CSV & Save to Vectorstore
df = pd.read_csv(courses_csv_path)
courses_texts = []
courses_metadata = []

df = pd.read_csv(courses_csv_path)
courses_texts = []
courses_metadata = []

for _, row in df.iterrows():
    text_data = ""
    metadata = {}

    for col in df.columns:
        value = str(row[col]) if not pd.isna(row[col]) else ""
        text_data += f"{col}: {value}. "
        metadata[col] = value

    courses_texts.append(text_data.strip())
    courses_metadata.append(metadata)

vectorstore_courses = Chroma.from_texts(
    texts=courses_texts,
    embedding=OpenAIEmbeddings(),
    metadatas=courses_metadata,
    persist_directory="chroma_courses"
)
vectorstore_courses.persist()


courses_retriever = vectorstore_courses.as_retriever(search_kwargs={"k": 4})

def extract_not_satisfied_sections(docs):
    all_text = "\n".join(doc.page_content for doc in docs)
    pattern = re.compile(
        r"(.*?)Not Satisfied.*?Units:\s*([\d.]+)\s*required,\s*([\d.]+)\s*used(?:,\s*([\d.]+)\s*needed)?",
        re.DOTALL | re.IGNORECASE
    )
    results = pattern.findall(all_text)

    extracted = []
    for match in results:
        category, required, used, needed = match
        extracted.append({
            "category": category.strip(),
            "required": float(required),
            "used": float(used),
            "needed": float(needed) if needed else round(float(required) - float(used), 2)
        })

    return extracted


#  Processing Functions
def process_whatif_report(user_question):
    # Step 1: Get requirement info
    requirement_docs = pdf_retriever.invoke("All unmet requirements and Not Satisfied sections in the What-If Report")
    filtered_requirements = "\n\n".join(
        doc.page_content for doc in requirement_docs if "Not Satisfied" in doc.page_content
    )
    if not filtered_requirements.strip():
        filtered_requirements = "\n\n".join(doc.page_content for doc in requirement_docs)

    # Step 2: Get Course History (likely on last page, so search directly)
    course_history_docs = pdf_retriever.invoke("Course History")
    course_history_text = "\n\n".join(doc.page_content for doc in course_history_docs)

    # Step 3: Full academic prompt
    pdf_prompt = [
        SystemMessage(content="You are a degree audit analyst reviewing a student's What-If report."),
        HumanMessage(content=(
            "You're reading a Penn State What-If Degree Audit PDF. Based on the document, give a complete academic breakdown with the following format:\n\n"
            "1. Summary of Academic Progress:\n"
            "- Show the student’s major/option, cumulative GPA.\n\n"
            "2. List of All Courses Taken from the Course History section below:\n"
            "- Include **both** completed and In-Progress courses.\n"
            "- Present as a table with: Term, Course Code, Course Title, Credits, and Grade (IP = in progress).\n\n"
            "3. Remaining Requirements:\n"
            "- Carefully analyze any section marked 'Not Satisfied'.\n"
            "- For each requirement category (such as General Education, Prescribed Courses, Additional Comp Sci/Math/Stat, Supporting Courses, etc.):\n"
            "  • State how many credits are required vs how many are completed.\n"
            "  • List the number of credits still needed.\n"
            "  • If specific courses or course types are mentioned as options to fulfill them, list them.\n"
            "  • Add the classes the student took in this category to the courses list above.\n"
            "  • Include whether a GPA condition is required for a category (like 2.5 in Prescribed CS courses).\n\n"
            f"Course History:\n{course_history_text}\n\n"
            f"Academic Requirements Context:\n{filtered_requirements}"
        ))
    ]

    # Step 4: Focused Not Satisfied summary prompt
    pdf_prompt2 = [
        SystemMessage(content="You are a degree audit analyst reviewing a student's What-If report."),
        HumanMessage(content=(
            "List **all** requirement categories marked as **Not Satisfied** in the academic audit report below.\n\n"
            "For each Not Satisfied category:\n"
            "- Give the full **requirement category name** (e.g., Prescribed Courses - General Option, Additional Comp Sci/Math/Stat, etc.) Do it for all the categories.\n"
            "- Report:\n"
            "  • Credits required\n"
            "  • Credits completed (or in progress if marked)\n"
            "  • Credits still needed\n"
            "  • Whether a GPA condition exists\n"
            "  • Whether any specific courses or course types are required\n\n"
            "At the end, give a **total count of all unsatisfied credits** by summing the missing credits from each Not Satisfied section.\n\n"
            f"Context:\n{filtered_requirements}"
        ))
    ]

    # Run both prompts
    summary = llm2.invoke(pdf_prompt).content.strip()
    summary2 = llm2.invoke(pdf_prompt2).content.strip()

    return summary + "\n\n---\n\n" + summary2





def process_psu_courses(user_question):
    relevant_courses = courses_retriever.invoke(user_question)
    if not relevant_courses:
        return "No relevant courses found."
    context_summary = "\n\n".join(doc.page_content for doc in relevant_courses)
    return context_summary





def aggregate_and_generate_response(user_question):
    print("\n🔹 Retrieving Student Record...")
    student_record = process_whatif_report(user_question)
    print("\n🔹 Retrieving Major Requirements...")
    major_requirement = process_psu_courses(user_question)



    # Check if the user is asking about prerequisites
    if any(word in user_question.lower() for word in
           ["prerequisite", "prereq", "prerequisites", "prerequists", "prerequist"]):
        G = course_graph

        course_code_match = re.search(r"[A-Z]{2,5}\s?[0-9]{2,3}[A-Z]?",user_question.upper())
        if course_code_match:
            course_code = course_code_match.group().replace("\xa0", " ").replace("  ", " ").strip()
            print("Extracted course code:", course_code)

            if course_code not in G:
                similar = [node for node in G.nodes if
                           course_code in node or course_code.replace(" ", "") in node.replace(" ", "")]
                return f"No course named '{course_code}' found.\nDid you mean: {similar[:3]}?"

            full_graph_text = G
            prereq_answer = get_course_prerequisites(course_code, G)
            return prereq_answer
        else:
            return "Please specify a valid course code like CMPSC 472."

    full_graph_text = course_graph

    instruction_prompt = SystemMessage(
        content=(
            "You are an academic advisor assistant AI.\n\n"
            "TASK 1: If the user asks about general education codes (e.g., GH, GS, CMPAB_BS), explain what the student has completed and still needs, using the context from their student record.\n\n"
            "TASK 2: If the user asks about a graduation plan, do the following:\n"
            f"1. Use the student's academic record ({student_record}) to find what courses/credits they've already satisfied, what courses or units or credits they still need to satisfy, and their major/option.\n"
            f"2. After retrieving all the student's records and major information from ({student_record}), Use the major requirements document({major_requirement}) to find the student's major,and all required courses for the student's (major and/or option) to graduate including general education requirments. Then\n"
            f"3. Subtract courses/units already completed and satisfied or in-progress (IP)(retrieved from {student_record}), from the required courses/units still needed for the major (retrieved from the major requirements {major_requirement}) And put them in a list, that list is the courses the student still needs to graduate.\n"
          #  f"4. Use the prerequisite graph ({full_graph_text}), to make sure courses are only recommended when prerequisites are met.\n"
            "4. Divide remaining courses (from the lsit) over available semesters (8 total max minus semesters the student already completed, will equal semsters the student have left, or you can just equally divide the courses into semsters with max of 15 credits per semster).\n"
            f"5. Structure the graduation plan like a bulletin semesters each semester includes the courses the student hasn't taken yet but needs to. ONLY OUTPUT COURSES NOT TAKEN YET, EXCLUDE COURSES TAKEN AND/OR IN-PROGRESS(IP)\n"
            f"6. Output a detailed accurate semester-by-semester graduation plan of class's/courses still needed to satisfy unsatisfied unit requirments, find the exact number that matches each course name from ({full_graph_text}, Example output for each course in the graduation plan, strictly follow that format : (CMPSC441: AI, prerequisites: CMPSC462). Get the values from {full_graph_text}) .\n"
            f"6. Give a summary of all the student's info at the end, and give some notes/advice. Mention missing information if any \n\n"
            f"TASK 3: If the user asks about prerequisites for a course, answer using the prerequisite graph ({full_graph_text})."
        )
    )
    final_prompt = [
        instruction_prompt,
        HumanMessage(
            content=f"User Question: {user_question}\nStudent's Academic Records: {student_record}\nMajor Requirements (bulletin): {major_requirement}\nAnswer the question or Generate a graduation plan")
    ]
    graduation_plan = llm1.invoke(final_prompt).content.strip()
    return f"**Graduation Plan:**\n{graduation_plan}"



####  MAIN PROGRAM ####
if __name__ == "__main__":

    print("\n Starting your AI academic Advisor...")
    user_question = input("Ask your question: ")
    print("\nProcessing your query...\n")
    try:
        result = aggregate_and_generate_response(user_question)
        print("\n Final Result:")
        print(result)
    except Exception as e:
        print(f"\n Error encountered: {e}")
