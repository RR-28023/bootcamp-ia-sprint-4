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

def get_prompt2(movie: Movie) -> str:
    prompt = f"Esta pelicula pertenece al género {movie.genre_tags}, es del año {movie.year} y su trama es: {movie.synopsis}"

    return prompt

def get_prompt3(movie: Movie) -> str:
    prompt = (f"Esta película pertenece al género {movie.genre_tags}, "
              f"es del año {movie.year}, "
              f"y su trama es: {movie.synopsis}. "
              f"Dirigida por: {movie.director_top_5}. "
              f"Guionistas: {movie.script_top_5}. "
              f"Protagonistas: {movie.cast_top_5}.")
    return prompt

# def ...
