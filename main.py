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

# ======================= LLM INITIALIZATION =======================
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

# ======================= SUMMARIZATION =======================
def summarize_text(text: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a summarizer of the highest quality and clarity."),
        HumanMessage(content="{text}")
    ])
    chain = prompt | llm
    return chain.invoke({"text": text}).content.strip()

# ======================= TRANSLATION (argostranslate) =======================
# Load translation models
try:
    argostranslate.package.update_package_index()
    argostranslate.translate.load_installed_packages()
except Exception as e:
    print(f"[ArgoTranslation Error] Failed to load packages: {e}")

def translate(text: str, target_lang: str) -> str:
    try:
        # Map standard lang codes to ArgoTranslate ones
        lang_map = {
            "hi": "hi", "bn": "bn", "ta": "ta",
            "te": "te", "ml": "ml", "kn": "kn"
        }
        if target_lang not in lang_map:
            return "[Translation Error] Unsupported language"

        from_lang = "en"
        to_lang = lang_map[target_lang]

        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang_obj = next((lang for lang in installed_languages if lang.code == from_lang), None)
        to_lang_obj = next((lang for lang in installed_languages if lang.code == to_lang), None)

        if not from_lang_obj or not to_lang_obj:
            return "[Translation Error] Languages not installed"

        translation = from_lang_obj.get_translation(to_lang_obj)
        return translation.translate(text)

    except Exception as e:
        return f"[Translation Error] {e}"

# ======================= TEXT-TO-SPEECH (gTTS) =======================
def text_to_speech(text: str, lang="hi") -> str:
    try:
        tts = gTTS(text, lang=lang)
        filename = f"{uuid.uuid4().hex}.mp3"
        path = os.path.join(tempfile.gettempdir(), filename)
        tts.save(path)
        return path
    except Exception as e:
        print(f"[TTS Error] {e}")
        return ""

# ======================= PDF PROCESSING + VECTOR DB =======================
def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings)
    retriever = vectordb.as_retriever()
    return chunks, retriever

# ======================= SUMMARIZE CHUNKS =======================
def summarize_chunks(chunks: List[str]) -> str:
    summaries = [summarize_text(doc.page_content) for doc in chunks[:10]]
    return "\n\n".join(summaries)

# ======================= RAG Q&A =======================
def answer_question(retriever, question: str) -> str:
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(question)
