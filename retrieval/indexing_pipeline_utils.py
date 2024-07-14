from __future__ import annotations
import nltk
from nltk.corpus import stopwords
import re
import spacy 

from langchain_core.documents import Document

from data_utils import Movie
from retrieval import config


def create_docs_to_embedd(movies: list[Movie], config: config.RetrievalExpsConfig) -> list[Document]:
    """
    Convierte una lista de objetos `Movie` a una lista the objetos `Document`(usada por Langchain).
    En esta función se decide que parte de los datos será usado como embeddings y que parte como metadata.
    """
    movies_as_docs = []
    for movie in movies:
        content = config.text_to_embed_fn(movie)
        metadata = movie.model_dump()
        doc = Document(page_content=content, metadata=metadata)
        movies_as_docs.append(doc)

    return movies_as_docs


## Posibles funciones para usar como `text_to_embed_fn` en `RetrievalExpsConfig` ##


def get_synopsys_txt(movie: Movie) -> str:
    return movie.synopsis

def get_synopsys_txt_first_test(movie: Movie) -> str:
    synopsis_first = movie.synopsis + movie.genre_tags + movie.title_es + movie.year
    return synopsis_first

def get_synopsys_txt_second_test(movie: Movie) -> str:
    synopsis_first = movie.synopsis + movie.genre_tags + movie.title_es
    return synopsis_first

def get_synopsys_txt_third_test(movie: Movie) -> str:
    
    nlp = spacy.load('es_core_news_sm')
    nltk.download('stopwords')

    synopsis = movie.synopsis + movie.genre_tags + movie.title_es

    synopsis = synopsis.strip().lower()

    # We drop stopwords.
    stop_words = set(stopwords.words('spanish'))
    synopsis_tokens = nltk.word_tokenize(synopsis)
    synopsis = ' '.join([word for word in synopsis_tokens if word not in stop_words])

    # We lemmatize the string.
    doc = nlp(synopsis)
    synopsis = ' '.join([token.lemma_ for token in doc])

    return synopsis

