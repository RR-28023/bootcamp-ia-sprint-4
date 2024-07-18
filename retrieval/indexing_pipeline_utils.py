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


def get_synopsys_and_genres_txt(movie: Movie) -> str:
    return f"{movie.synopsis}; {movie.genre_tags}"

def get_synopsis_genre_country_and_director(movie: Movie) -> str:
    return f"{movie.synopsis}; {movie.genre_tags}; {movie.country}; {movie.director_top_5}"

def get_synopsis_genre_and_country(movie: Movie) -> str:
    return f"{movie.synopsis}; {movie.genre_tags}; {movie.country}"

def get_synopsis_genre_and_director(movie: Movie) -> str:
    return f"{movie.synopsis}; {movie.genre_tags}; {movie.director_top_5}"