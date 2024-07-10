from __future__ import annotations

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

# def ...

def get_synopsys_genre_txt(movie: Movie) -> str:

    prompt = movie.genre_tags.lower() + " en el que " + movie.synopsis
    prompt = prompt.replace("(FILMAFFINITY)", "").strip()

    return prompt

def get_synopsys_genre_2_txt(movie: Movie) -> str:

    if ';' in movie.genre_tags:
        movie_genre = movie.genre_tags.split(';', 1)[0]
        movie_genre_1 = movie.genre_tags.split(';', 1)[1]
    else:
        movie_genre = movie.genre_tags
        movie_genre_1 = ""

    prompt = movie_genre + " " + movie_genre_1 + " en el que " + movie.synopsis
    prompt = prompt.replace("(FILMAFFINITY)", "").strip()

    return prompt