from google import genai

client = genai.Client(api_key="YOUR_API_KEY")


def generate_basic_answer(question):
    prompt = f"""
Answer the following question using only your own knowledge.
Do not search the web or use external sources.

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text
