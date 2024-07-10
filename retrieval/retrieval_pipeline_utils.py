import re


def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query


def new_clean_query_txt(query: str) -> str:
    file = open("/Users/vladimirmaksimov/Dev/Todo/bootcamp-ia-sprint-4/retrieval/data/spanish.txt", encoding="utf8")
    stop_words = []
    for stop_word in file:
        stop_words.append(str(stop_word.strip()))
    query = query.replace("El usuario busca ", "").strip()
    # Delete special characters, numbers and punctuation.
    query = re.sub(r'[^A-Za-z\s, á, é, ú, ü, í, ó, ñ]', '', query)
    query = re.sub(r'[,]+' , '' , query)
    # Replace upper letters with lower ones.
    query = query.lower()
    # Split the query into the list of words.
    query_list = list(query.split())
    # Join the words into a phrase without stop words.
    query = ' '.join([word for word in query_list if word not in stop_words])
    return query
