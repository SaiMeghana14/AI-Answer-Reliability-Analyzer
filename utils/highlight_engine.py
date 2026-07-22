import re


class HighlightEngine:

    @staticmethod
    def highlight(
        answer,
        evidence
    ):

        evidence_words = set(

            re.findall(
                r"\w+",
                evidence.lower()
            )

        )

        html = ""

        for word in answer.split():

            clean = re.sub(
                r"[^\w]",
                "",
                word.lower()
            )

            if clean in evidence_words:

                html += (
                    f"<span style='"
                    "background:#d4edda;"
                    "padding:3px;"
                    "border-radius:4px;'>"
                    f"{word}</span> "
                )

            else:

                html += (
                    f"<span style='"
                    "background:#f8d7da;"
                    "padding:3px;"
                    "border-radius:4px;'>"
                    f"{word}</span> "
                )

        return html
