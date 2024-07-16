# import nltk
def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    # Remove stopwords from query
    # stopwords = nltk.corpus.stopwords.words("spanish")
    # query = " ".join([word for word in query.split() if word not in stopwords])
    return query
