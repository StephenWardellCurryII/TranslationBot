import streamlit as st
from main import summarize_text, translate, text_to_speech, process_pdf, summarize_chunks, answer_question

LANG_MAP = {
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn"
}

st.set_page_config(page_title="Multilingual AI Summarizer", layout="centered")

st.title("📄 Multilingual AI Summarizer with Q&A")

native_lang = st.selectbox("Select your native language:", list(LANG_MAP.keys()))

option = st.radio("Choose input type:", ["Text", "PDF Document"])

if option == "Text":
    user_text = st.text_area("Enter the text to summarize:")
    if st.button("Generate Summary"):
        if user_text:
            summary = summarize_text(user_text)
            translated = translate(summary, LANG_MAP[native_lang])
            st.subheader("Summarized Text:")
            st.success(translated)
            audio_path = text_to_speech(translated)
            st.audio(audio_path, format="audio/wav")

            question = st.text_input("Ask a question about the summary:")
            if question:
                translated_q = translate(question, "en")
                ans = summarize_text(user_text + "\n\nQuestion: " + translated_q)
                translated_a = translate(ans, LANG_MAP[native_lang])
                st.subheader("Answer:")
                st.success(translated_a)
                audio_ans_path = text_to_speech(translated_a)
                st.audio(audio_ans_path, format="audio/wav")
        else:
            st.warning("Please enter some text.")

elif option == "PDF Document":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file and st.button("Process PDF"):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        chunks, retriever = process_pdf("temp.pdf")
        summary = summarize_chunks(chunks)
        translated = translate(summary, LANG_MAP[native_lang])
        st.subheader("Summarized PDF:")
        st.success(translated)
        audio_path = text_to_speech(translated)
        st.audio(audio_path, format="audio/wav")

        question = st.text_input("Ask a question about the PDF:")
        if question:
            translated_q = translate(question, "en")
            ans = answer_question(retriever, translated_q)
            translated_a = translate(ans, LANG_MAP[native_lang])
            st.subheader("Answer:")
            st.success(translated_a)
            audio_ans_path = text_to_speech(translated_a)
            st.audio(audio_ans_path, format="audio/wav")
