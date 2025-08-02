# --- app.py (Frontend) ---

import os
import streamlit as st
from main import summarize_text, translate, text_to_speech, process_pdf, summarize_chunks, answer_question, answer_from_text

LANG_MAP = {
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Bengali": "bn"
}

st.set_page_config(page_title="Multilingual Summarizer", layout="centered")
st.title("Multilingual AI Summarizer & Q&A")

language = st.selectbox("Choose your language:", LANG_MAP.keys())
lang_code = LANG_MAP[language]

input_type = st.radio("Select input type:", ["Text", "PDF Document"])

# =============== TEXT INPUT ===================== #
if input_type == "Text":
    user_input = st.text_area("Enter the text to summarize:",key="summary_input")
    st.write("DEBUG: User input:", repr(user_input))


    if st.button("Generate Summary"):
        input_text=st.session_state.get("summary_input", "")
        st.write("DEBUG: Current session_state input:", repr(st.session_state.get("summary_input", "")))
        if user_input.strip():
            summary_en = summarize_text(input_text)
            st.write("DEBUG: English Summary:", repr(summary_en))
            summary_local = translate(summary_en, lang_code)

            st.subheader("Summarized Text:")
            st.success(summary_local)

            audio_file = text_to_speech(summary_local, lang=lang_code)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")

            question = st.text_input("Ask a question about the summary:")
            if question.strip():
                if not user_input.strip():
                    st.warning("Please enter text before asking a question.")
                else:
                    translated_q = translate(question, "en")
                    answer_en = answer_from_text(user_input, translated_q)
                    answer_local = translate(answer_en, lang_code)

                    st.subheader("Answer:")
                    st.success(answer_local)

                    audio_ans = text_to_speech(answer_local, lang=lang_code)
                    if audio_ans:
                        st.audio(audio_ans, format="audio/mp3")
        else:
            st.warning("Please enter some text to summarize.")

# =============== PDF INPUT ===================== #
else:
    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_pdf and st.button("Process PDF"):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_pdf.read())

        chunks, retriever = process_pdf("temp.pdf")
        os.remove("temp.pdf")

        summary_en = summarize_chunks(chunks)
        summary_local = translate(summary_en, lang_code)

        st.subheader("Summarized PDF:")
        st.success(summary_local)

        audio_file = text_to_speech(summary_local, lang=lang_code)
        if audio_file:
            st.audio(audio_file, format="audio/mp3")

        question = st.text_input("Ask a question about the PDF:")
        if question.strip():
            translated_q = translate(question, "en")
            answer_en = answer_question(retriever, translated_q)
            answer_local = translate(answer_en, lang_code)

            st.subheader("Answer:")
            st.success(answer_local)

            audio_ans = text_to_speech(answer_local, lang=lang_code)
            if audio_ans:
                st.audio(audio_ans, format="audio/mp3")
