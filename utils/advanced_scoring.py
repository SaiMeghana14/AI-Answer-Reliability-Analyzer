import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# Semantic Similarity
# -------------------------
def semantic_similarity(a, b):
    vect = TfidfVectorizer().fit_transform([a, b])
    sim = cosine_similarity(vect[0:1], vect[1:2])[0][0]
    return float(sim)


# -------------------------
# Entity Extraction
# (simple heuristic version)
# -------------------------
def extract_entities(text):

    dates = re.findall(r'\b(18\d{2}|19\d{2}|20\d{2})\b', text)

    numbers = re.findall(r'\b\d+\b', text)

    caps = re.findall(r'\b[A-Z][a-z]+\b', text)

    return set(dates + numbers + caps)


def entity_consistency(answer, reference):

    a = extract_entities(answer)
    r = extract_entities(reference)

    if len(r)==0:
        return 0.5

    overlap = len(a.intersection(r))
    return overlap / len(r)


# -------------------------
# Contradiction detection
# (simple number/date conflicts)
# -------------------------
def contradiction_penalty(answer, reference):

    a_nums = set(re.findall(r'\b\d+\b', answer))
    r_nums = set(re.findall(r'\b\d+\b', reference))

    if not a_nums or not r_nums:
        return 0

    if a_nums != r_nums:
        return 1

    return 0


# -------------------------
# Citation support
# crude overlap proxy
# -------------------------
def citation_support(answer, reference):
    return semantic_similarity(answer, reference)


# -------------------------
# Final Multi-factor Score
# -------------------------
def reliability_score(answer, reference):

    sim = semantic_similarity(answer, reference)

    cite = citation_support(answer, reference)

    entity = entity_consistency(answer, reference)

    contradiction = contradiction_penalty(answer, reference)

    score = (
        0.4*sim +
        0.3*cite +
        0.2*entity -
        0.1*contradiction
    )

    return max(score,0)*100
