import nltk
from nltk.corpus import stopwords
import re
import spacy 


def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

def clean_query_txt_first_test(query: str) -> str:

    nlp = spacy.load('es_core_news_sm')
    nltk.download('stopwords')

    query = query.replace("El usuario busca ", "").strip().lower()

    # We remove extra spaces and line breaks.
    query = re.sub(r'[ ]+', ' ', query)
    query = re.sub(r'\n', ' ', query)

    # We remove all characters that are not letters, numbers or whitespace.
    query = re.sub(r'[^a-z0-9áéíóúüñ\s]', '', query)

    # We drop stopwords.
    stop_words = set(stopwords.words('spanish'))
    query_tokens = nltk.word_tokenize(query)
    query = ' '.join([word for word in query_tokens if word not in stop_words])

    # We lemmatize the string.
    doc = nlp(query)
    query = ' '.join([token.lemma_ for token in doc])

    return query

def clean_query_txt_second_test(query: str) -> str:

    nlp = spacy.load('es_core_news_sm')
    nltk.download('stopwords')

    query = query.replace("El usuario busca ", "").strip().lower()

    # We drop stopwords.
    stop_words = set(stopwords.words('spanish'))
    query_tokens = nltk.word_tokenize(query)
    query = ' '.join([word for word in query_tokens if word not in stop_words])

    # We lemmatize the string.
    doc = nlp(query)
    query = ' '.join([token.lemma_ for token in doc])

    return query


def clean_query_txt_third_test(query: str) -> str:

    nlp = spacy.load('es_core_news_sm')
    nltk.download('stopwords')

    query = query.replace("El usuario busca ", "").strip().lower()

    # We remove extra spaces and line breaks.
    query = re.sub(r'[ ]+', ' ', query)
    query = re.sub(r'\n', ' ', query)

    # We remove all characters that are not letters, numbers or whitespace.
    query = re.sub(r'[^a-z0-9áéíóúüñ\s]', '', query)

    # We drop stopwords.
    stop_words = set(stopwords.words('spanish'))
    query_tokens = nltk.word_tokenize(query)
    query = ' '.join([word for word in query_tokens if word not in stop_words])

    # We lemmatize the string.
    doc = nlp(query)
    query = ' '.join([token.lemma_ for token in doc])

    return query


def clean_query_txt_fourth_test(query: str) -> str:

    query = query.replace("El usuario busca ", "").strip().lower()

    # We remove extra spaces and line breaks.
    query = re.sub(r'[ ]+', ' ', query)
    query = re.sub(r'\n', ' ', query)

    # We remove all characters that are not letters, numbers or whitespace.
    query = re.sub(r'[^a-z0-9áéíóúüñ\s]', '', query)

    return query