import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')


def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

def clean_query_txt_v2(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()

    # Convertir a minúsculas
    query = query.lower()

    # Devolvemos la query
    return query

def clean_query_txt_v3(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()

    # Convertir a minúsculas
    query = query.lower()

    # Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    query_tokens = nltk.word_tokenize(query)
    query = ' '.join([word for word in query_tokens if word not in stop_words])

    # Devolvemos la query
    return query

