def find_missing_info(ai_answer, real_data):
    ai_words = set(ai_answer.lower().split())
    real_words = set(real_data.lower().split())

    missing = real_words - ai_words
    return list(missing)[:15]


def find_overlap(ai_answer, real_data):
    ai_words = set(ai_answer.lower().split())
    real_words = set(real_data.lower().split())

    overlap = ai_words.intersection(real_words)
    return list(overlap)[:15]

def highlight_text(hallucinations):
    highlighted = ""
    for item in hallucinations:
        sentence = item["sentence"]
        if item["hallucinated"]:
            highlighted += f"❌ {sentence}\n\n"
        else:
            highlighted += f"✅ {sentence}\n\n"
    return highlighted
