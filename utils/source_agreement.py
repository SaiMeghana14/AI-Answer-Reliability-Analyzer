from collections import Counter


class SourceAgreement:

    @staticmethod
    def confidence(sources):

        available = []

        for source in sources.values():

            if len(source["text"]) > 40:

                available.append(source["text"])

        count = len(available)

        if count >= 3:

            return "High"

        elif count == 2:

            return "Medium"

        elif count == 1:

            return "Low"

        else:

            return "Insufficient Evidence"

    @staticmethod
    def merged_text(sources):

        text = []

        for source in sources.values():

            if source["text"]:

                text.append(source["text"])

        return "\n\n".join(text)
