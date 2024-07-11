from __future__ import annotations

from langchain_core.documents import Document

import re
import nltk
from nltk.corpus import stopwords

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
    texto = ""

    if(movie.tv_show_flag):
        texto += "La serie"
    else:
        texto += "La película"

    texto += f" relacionada con los géneros {movie.genre_tags}"
    texto += f" tiene por sinopsis: {movie.synopsis}"
    return texto

def get_synopsys_txt_clean(movie: Movie) -> str:
    texto = ""

    if(movie.tv_show_flag):
        texto += "La serie"
    else:
        texto += "La película"

    texto += f" con los géneros {movie.genre_tags.replace(';', ',')}"
    texto += f" y sinopsis: {clean_txt(movie.synopsis)}"
    return texto

def clean_txt(movie_property: str) -> str:
    clean_property = movie_property.lower()
    clean_property = re.sub(r'[^a-záéíóúñ\s]', '', clean_property)

    tokens = nltk.word_tokenize(clean_property)
    
    spanish_stopwords = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in spanish_stopwords]

    return ' '.join(tokens)


# def ...
