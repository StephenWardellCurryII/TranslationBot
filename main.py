import os
from typing import List
from langchain_community.chat_models import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import tempfile
import uuid
import numpy as np
import scipy.io.wavfile
from bark import SAMPLE_RATE, generate_audio

# Init LLM
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

# Summarization
def summarize_text(text: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a summarizer of the highest quality and clarity."),
        HumanMessage(content="{text}")
    ])
    chain = prompt | llm
    return chain.invoke({"text": text}).content.strip()

# Translator init
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

def translate(text: str, target_lang: str) -> str:
    tokenizer.src_lang = "en"
    encoded = tokenizer(text, return_tensors="pt")
    forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
    generated = model.generate(**encoded, forced_bos_token_id=forced_bos_token_id)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# TTS using Bark
def text_to_speech(text: str) -> str:
    audio_array = generate_audio(text)
    filename = f"{uuid.uuid4().hex}.wav"
    path = os.path.join(tempfile.gettempdir(), filename)
    scipy.io.wavfile.write(path, rate=SAMPLE_RATE, data=audio_array)
    return path

# PDF summarization and RAG setup
def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings)
    retriever = vectordb.as_retriever()
    return chunks, retriever

def summarize_chunks(chunks: List[str]) -> str:
    joined_text = " ".join([doc.page_content for doc in chunks])
    return summarize_text(joined_text)

def answer_question(retriever, question: str) -> str:
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(question)
