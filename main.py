# --- main.py (Backend) ---

import os
import tempfile
import uuid
from typing import List
from gtts import gTTS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import argostranslate.package
import argostranslate.translate

load_dotenv()

# LLM SETUP
llm = ChatGroq(temperature=0.3, model_name="llama3-8b-8192")

# Load translation packages
def load_argotranslate_models():
    try:
        argostranslate.package.update_package_index()
        argostranslate.translate.load_installed_packages()
    except Exception as e:
        print(f"[ArgoTranslate Error] {e}")

load_argotranslate_models()

# Summarize text using LLM
def summarize_text(text: str) -> str:
    if not text.strip():
        return "[Error] No text provided for summarization"

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="Summarize the following content clearly and concisely:"),
            HumanMessage(content="{text}")
        ],
    )

    chain = prompt | llm

    try:
        result = chain.invoke({"text": text})
        return result.content.strip()
    except Exception as e:
        return f"[Error] Summarization failed: {e}"


# New QA function for text
def answer_from_text(text: str, question: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You're a helpful assistant answering questions from a given text."),
        HumanMessage(content="Context:\n{text}\n\nQuestion:\n{question}")
    ])
    chain = prompt | llm
    return chain.invoke({"text": text, "question": question}).content.strip()

# Translate using Argo
def translate(text: str, target_lang: str) -> str:
    try:
        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next((l for l in installed_languages if l.code == "en"), None)
        to_lang = next((l for l in installed_languages if l.code == target_lang), None)

        if not from_lang or not to_lang:
            return "[Translation Error] Required languages not installed"

        translation = from_lang.get_translation(to_lang)
        return translation.translate(text)

    except Exception as e:
        return f"[Translation Error] {e}"

# Text-to-speech
def text_to_speech(text: str, lang="hi") -> str:
    try:
        tts = gTTS(text=text, lang=lang)
        filename = f"{uuid.uuid4().hex}.mp3"
        path = os.path.join(tempfile.gettempdir(), filename)
        tts.save(path)
        return path
    except Exception as e:
        print(f"[TTS Error] {e}")
        return ""

# PDF processing
def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)
    return chunks, vectordb.as_retriever()

# Summarize multiple chunks
def summarize_chunks(chunks: List[str]) -> str:
    return "\n\n".join(summarize_text(doc.page_content) for doc in chunks[:5])

# Retrieval QA for PDF
def answer_question(retriever, question: str) -> str:
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(question)