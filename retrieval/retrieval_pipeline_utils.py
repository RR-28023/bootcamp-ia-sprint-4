import re
import nltk
from nltk.corpus import stopwords

def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    query = query.lower()

    # Eliminar caracteres especiales, números, and signos de puntuación
    query = re.sub(r'[^a-záéíóúñ\s]', '', query)

    return query

def clean_stopwords_query_txt(query: str) -> str:
    query = clean_query_txt(query)

    tokens = nltk.word_tokenize(query)
    
    spanish_stopwords = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in spanish_stopwords]

    return ' '.join(tokens)