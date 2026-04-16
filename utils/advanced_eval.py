import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------- Sentence Split (NO NLTK) --------
def sentence_split(text):
    # Split by ., ?, !
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


# -------- Similarity --------
def sentence_similarity(sent, reference):
    vectorizer = TfidfVectorizer().fit_transform([sent, reference])
    sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
    return float(sim[0][0])


# -------- Hallucination Detection --------
def detect_hallucinations(answer, reference, threshold=0.3):
    sentences = sentence_split(answer)

    results = []
    for sent in sentences:
        sim = sentence_similarity(sent, reference)
        results.append({
            "sentence": sent,
            "similarity": sim,
            "hallucinated": sim < threshold
        })

    return results


# -------- Explain Winner --------
def explain_winner(score1, score2, name1="Basic AI", name2="OpenAI"):
    diff = abs(score1 - score2)

    winner = name1 if score1 > score2 else name2

    if diff > 20:
        reason = "Significantly higher factual overlap with trusted data."
    elif diff > 10:
        reason = "Moderately better alignment with retrieved knowledge."
    else:
        reason = "Slight improvement in semantic similarity."

    return winner, reason
