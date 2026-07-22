import wikipedia
import requests
from urllib.parse import quote


class RetrievalEngine:

    def __init__(self):

        self.sources = {}

    # ------------------------------------
    # Wikipedia
    # ------------------------------------

    def wikipedia(self, query):

        try:

            summary = wikipedia.summary(
                query,
                sentences=4,
                auto_suggest=False
            )

            self.sources["Wikipedia"] = {
                "text": summary,
                "url": f"https://en.wikipedia.org/wiki/{quote(query)}"
            }

        except:

            self.sources["Wikipedia"] = {
                "text": "",
                "url": ""
            }

    # ------------------------------------
    # DuckDuckGo
    # ------------------------------------

    def duckduckgo(self, query):

        try:

            url = (
                f"https://api.duckduckgo.com/"
                f"?q={quote(query)}&format=json"
            )

            data = requests.get(
                url,
                timeout=6
            ).json()

            text = (
                data.get("AbstractText")
                or (
                    data.get("RelatedTopics", [{}])[0].get(
                        "Text", ""
                    )
                    if data.get("RelatedTopics")
                    else ""
                )
            )

            self.sources["DuckDuckGo"] = {

                "text": text,

                "url":
                f"https://duckduckgo.com/?q={quote(query)}"

            }

        except:

            self.sources["DuckDuckGo"] = {

                "text": "",

                "url": ""

            }

    # ------------------------------------
    # PubMed (placeholder)
    # ------------------------------------

    def pubmed(self, query):

        self.sources["PubMed"] = {

            "text": "",

            "url":
            f"https://pubmed.ncbi.nlm.nih.gov/?term={quote(query)}"

        }

    # ------------------------------------
    # arXiv (placeholder)
    # ------------------------------------

    def arxiv(self, query):

        self.sources["arXiv"] = {

            "text": "",

            "url":
            f"https://arxiv.org/search/?query={quote(query)}"

        }

    # ------------------------------------
    # Retrieve everything
    # ------------------------------------

    def retrieve(self, query):

        self.sources = {}

        self.wikipedia(query)

        self.duckduckgo(query)

        self.pubmed(query)

        self.arxiv(query)

        return self.sources
