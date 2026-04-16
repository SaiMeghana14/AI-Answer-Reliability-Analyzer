from openai import OpenAI
import os
import time

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_openai_answer(question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        if "rate_limit" in str(e).lower():
            return "⚠️ OpenAI rate limit reached. Showing fallback answer."
        return f"Error: {str(e)}"
