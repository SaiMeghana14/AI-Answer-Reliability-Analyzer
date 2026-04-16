from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(ai_answer, real_data):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([ai_answer, real_data])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return float(similarity[0][0])


def get_score(similarity):
    score = similarity * 100

    if score > 75:
        confidence = "High Confidence ✅"
    elif score > 50:
        confidence = "Moderate Confidence ⚠️"
    else:
        confidence = "Low Confidence ❌"

    return round(score, 2), confidence
