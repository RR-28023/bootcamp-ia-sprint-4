from __future__ import annotations

from langchain_core.documents import Document

from data_utils import Movie
from retrieval import config

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')


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

def get_synopsys_txt_v2(movie: Movie) -> str:
    # Devolvemos la sinopsis y el título
    return f'{movie.synopsis} \n {movie.title_es}'

def get_synopsys_txt_v3(movie: Movie) -> str:
    # Convertir a minúsculas
    plot = movie.synopsis.lower()
    title = movie.title_es.lower()

    # Devolvemos la sinopsis y el título
    return f'{plot} \n {title}'

def get_synopsys_txt_v4(movie: Movie) -> str:
    # Convertir a minúsculas
    plot = movie.synopsis.lower()
    title = movie.title_es.lower()
    genres = movie.genre_tags.lower()

    # Devolvemos la sinopsis, el título y los géneros
    return f'{plot} \n {title} \n {genres}'

def get_synopsys_txt_v5(movie: Movie) -> str:
    # Convertir a minúsculas
    plot = movie.synopsis.lower()
    title = movie.title_es.lower()
    genres = movie.genre_tags.lower()

    # Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    plot_tokens = nltk.word_tokenize(plot)
    title_tokens = nltk.word_tokenize(title)
    genres_tokens = nltk.word_tokenize(genres)
    plot = ' '.join([word for word in plot_tokens if word not in stop_words])
    title = ' '.join([word for word in title_tokens if word not in stop_words])
    genres = ' '.join([word for word in genres_tokens if word not in stop_words])

    # Devolvemos la sinopsis, el título y los géneros
    return f'{plot} \n {title} \n {genres}'
