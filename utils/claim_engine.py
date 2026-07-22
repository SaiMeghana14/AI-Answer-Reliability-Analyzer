import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ClaimEngine:

    @staticmethod
    def extract_claims(text):

        claims = re.split(r'[.!?]+', text)

        claims = [
            c.strip()
            for c in claims
            if len(c.strip()) > 8
        ]

        return claims


    @staticmethod
    def similarity(a, b):

        tfidf = TfidfVectorizer()

        matrix = tfidf.fit_transform(
            [a, b]
        )

        return cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]


    @classmethod
    def verify_claim(
        cls,
        claim,
        evidence
    ):

        score = cls.similarity(
            claim,
            evidence
        )

        if score >= 0.70:

            label = "Supported"

            icon = "✅"

            confidence = "High"

        elif score >= 0.40:

            label = "Partially Supported"

            icon = "⚠️"

            confidence = "Medium"

        else:

            label = "Unsupported"

            icon = "❌"

            confidence = "Low"

        return {

            "claim": claim,

            "score": round(
                score*100,
                2
            ),

            "label": label,

            "icon": icon,

            "confidence": confidence

        }


    @classmethod
    def analyze(
        cls,
        answer,
        evidence
    ):

        claims = cls.extract_claims(
            answer
        )

        results = []

        for claim in claims:

            results.append(

                cls.verify_claim(
                    claim,
                    evidence
                )

            )

        return results
