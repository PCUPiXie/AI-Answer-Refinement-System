import streamlit as st
import google.generativeai as genai
from groq import Groq
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# configure models
genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------
# GEMINI
# ---------------------------

def ask_gemini(question):

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(question)

    return response.text


# ---------------------------
# GROQ
# ---------------------------

def ask_groq(question):

    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": question}]
    )

    return response.choices[0].message.content


# ---------------------------
# OPENAI
# ---------------------------

def ask_openai(question):

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )

    return response.choices[0].message.content


# ---------------------------
# REFINER
# ---------------------------

def refine_answer(question, a1, a2, a3):

    prompt = f"""
    Question: {question}

    These are answers from 3 AI models.

    Gemini:
    {a1}

    Groq:
    {a2}

    OpenAI:
    {a3}

    Combine the best information and produce
    a clear and accurate final answer.
    """

    return ask_openai(prompt)


# ---------------------------
# STREAMLIT UI
# ---------------------------

st.title("AI Answer Refiner (Multi-LLM System)")

question = st.text_input("Ask a question")

if st.button("Generate Answer"):

    with st.spinner("Consulting multiple AIs..."):

        gemini_answer = ask_gemini(question)
        groq_answer = ask_groq(question)
        openai_answer = ask_openai(question)

        final_answer = refine_answer(
            question,
            gemini_answer,
            groq_answer,
            openai_answer
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Gemini")
        st.write(gemini_answer)

    with col2:
        st.subheader("Groq")
        st.write(groq_answer)

    with col3:
        st.subheader("OpenAI")
        st.write(openai_answer)

    st.subheader("Final Refined Answer")
    st.success(final_answer)
