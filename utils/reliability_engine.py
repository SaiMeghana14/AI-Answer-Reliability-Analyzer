import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ReliabilityEngine:

    # -------------------------
    # Semantic Similarity
    # -------------------------

    @staticmethod
    def semantic_similarity(answer, reference):

        tfidf = TfidfVectorizer()

        matrix = tfidf.fit_transform(
            [answer, reference]
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return round(score * 100, 2)

    # -------------------------
    # Entity Extraction
    # -------------------------

    @staticmethod
    def extract_entities(text):

        entities = set()

        # Years
        years = re.findall(
            r"\b(18\d{2}|19\d{2}|20\d{2})\b",
            text
        )

        entities.update(years)

        # Numbers
        nums = re.findall(
            r"\b\d+\b",
            text
        )

        entities.update(nums)

        # Capitalized words
        caps = re.findall(
            r"\b[A-Z][a-z]+\b",
            text
        )

        entities.update(caps)

        return entities

    # -------------------------
    # Entity Consistency
    # -------------------------

    @classmethod
    def entity_consistency(cls,
                           answer,
                           reference):

        a = cls.extract_entities(answer)

        r = cls.extract_entities(reference)

        if len(r) == 0:

            return 100

        overlap = len(
            a.intersection(r)
        )

        score = overlap / len(r)

        return round(score * 100, 2)

    # -------------------------
    # Citation Support
    # -------------------------

    @staticmethod
    def citation_support(answer,
                         reference):

        words = set(
            answer.lower().split()
        )

        ref = set(
            reference.lower().split()
        )

        overlap = len(
            words.intersection(ref)
        )

        if len(ref) == 0:

            return 0

        return round(
            overlap / len(ref) * 100,
            2
        )

    # -------------------------
    # Contradiction Detection
    # -------------------------

    @staticmethod
    def contradiction(answer,
                      reference):

        nums_a = set(
            re.findall(
                r"\b\d+\b",
                answer
            )
        )

        nums_r = set(
            re.findall(
                r"\b\d+\b",
                reference
            )
        )

        if len(nums_r) == 0:

            return 0

        if nums_a != nums_r:

            return 100

        return 0

    # -------------------------
    # Final Reliability
    # -------------------------

    @classmethod
    def evaluate(cls,
                 answer,
                 reference):

        semantic = cls.semantic_similarity(
            answer,
            reference
        )

        citation = cls.citation_support(
            answer,
            reference
        )

        entity = cls.entity_consistency(
            answer,
            reference
        )

        contradiction = cls.contradiction(
            answer,
            reference
        )

        final = (

            0.40 * semantic +

            0.30 * citation +

            0.20 * entity -

            0.10 * contradiction

        )

        final = max(0, final)

        return {

            "semantic": semantic,

            "citation": citation,

            "entity": entity,

            "contradiction": contradiction,

            "final": round(final, 2)

        }
