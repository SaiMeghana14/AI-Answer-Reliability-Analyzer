import textwrap

def generate_basic_answer(question):
    """
    Generates a baseline AI response without using any external API.
    This serves as the comparison model for TrustEval.
    """

    question = question.strip()

    return textwrap.dedent(f"""
    Question:
    {question}

    This response is generated using the model's built-in knowledge only,
    without retrieval or external evidence.

    The answer may provide a general explanation of the topic, but it can
    contain outdated information, missing details, or unsupported claims.
    Users should verify important information using reliable sources.

    This baseline response is intended for comparison with the
    Retrieval-Augmented AI response.
    """).strip()
