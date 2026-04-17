from utils.retriever import get_wiki_data

def generate_retrieval_answer(question):
    data = get_wiki_data(question)

    if data == "No reliable data found.":
        return "No reliable information available."

    # Simulate "AI reasoning"
    return f"Based on available knowledge: {data}"
