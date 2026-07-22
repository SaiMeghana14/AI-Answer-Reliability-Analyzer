import streamlit as st
from google import genai

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


def generate_basic_answer(question):
    """
    Generates a baseline answer using Gemini only
    (without retrieval).
    """

    prompt = f"""
Answer the following question using only your own knowledge.
Do not search the web or use external sources.

Question:
{question}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"Gemini Error: {e}"
