import openai

openai.api_key = "YOUR_API_KEY"

def generate_openai_answer(question):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )
    return response['choices'][0]['message']['content']
