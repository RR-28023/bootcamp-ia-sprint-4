def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

def clean_query_stopwords_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    stop_words = set(stopwords.words('spanish'))
    tokens = nltk.word_tokenize(query)
    query = ' '.join([word for word in tokens if word not in stop_words])
    return query