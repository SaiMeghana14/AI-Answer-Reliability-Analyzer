import wikipedia

def get_wiki_data(query):
    try:
        return wikipedia.summary(query, sentences=5)
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=5)
    except:
        return "No reliable data found."
