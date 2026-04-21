import requests
import wikipedia


def wiki_source(q):
    try:
        return wikipedia.summary(q, sentences=2)
    except:
        return ""


def duck_source(q):
    try:
        url=f"https://api.duckduckgo.com/?q={q}&format=json"
        j=requests.get(url).json()
        return j.get("AbstractText","")
    except:
        return ""


# Placeholder stubs
def pubmed_source(q):
    return ""


def arxiv_source(q):
    return ""


def news_source(q):
    return ""


def retrieve_all(q):

    sources = {
        "wiki": wiki_source(q),
        "duck": duck_source(q),
        "pubmed": pubmed_source(q),
        "arxiv": arxiv_source(q),
        "news": news_source(q)
    }

    return sources


def source_agreement(sources):

    non_empty = [
        v for v in sources.values()
        if len(v.strip())>20
    ]

    n=len(non_empty)

    if n>=3:
        return "High confidence"

    if n==2:
        return "Medium confidence"

    if n==1:
        return "Low confidence"

    return "Insufficient evidence"
