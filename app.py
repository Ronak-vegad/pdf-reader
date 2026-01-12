import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ------------------ CONFIG ------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") # Ensure it's in environment
FAISS_DIR = "faiss_index"

# ------------------ FUNCTIONS ------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted: text += extracted
    return text

def get_text_chunks(text):
    # Optimized for RAG precision
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200
    )
    return splitter.split_text(text)

def chunks_to_vector(text_chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_DIR)

def load_vector_store():
    if not os.path.exists(f"{FAISS_DIR}/index.faiss"):
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

def get_rag_chain():
    prompt = PromptTemplate(
        template="""Answer the question using ONLY the provided context. 
If the answer is not in the context, say "Answer is not provided in the context."

Context: {context}
Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # Fixed: Directly passing the input dictionary to the prompt
    return prompt | model | StrOutputParser()

def user_input(user_question):
    db = load_vector_store()
    if db is None:
        st.error("Please upload and process PDFs before asking questions.")
        return

    # Similarity search
    docs = db.similarity_search(user_question, k=5) # Retrieve top 5 chunks
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = get_rag_chain()
    # Corrected invocation
    response = chain.invoke({"context": context, "question": user_question})

    st.write("### Reply")
    st.write(response)

# ------------------ STREAMLIT FRONTEND ------------------
def main():
    st.set_page_config(page_title="PDF Chat using Gemini", layout="wide")
    st.header("📄 Chat with PDF")

    user_question = st.text_input("Ask a question from the uploaded PDFs")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                chunks_to_vector(text_chunks)
                st.success("PDFs processed successfully!")

if __name__ == "__main__":
    main()