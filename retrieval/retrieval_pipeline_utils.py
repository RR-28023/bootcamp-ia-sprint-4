

def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

def clean_low_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").lower().strip()
    return query

import nltk 
from nltk.corpus import stopwords 
nltk.download('stopwords') 
stop_words = set(stopwords.words('spanish')) 

def clean_query_2_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip().split()
    
    clean_query = []
    for word in query:
        if word not in stop_words:
            clean_query.append(word)

    clean_query = " ".join(clean_query)

    return clean_query


