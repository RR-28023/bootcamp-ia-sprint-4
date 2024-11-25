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

# Vamos a concatenar diferentes atributos para una mejor busqueda.

def get_detailed_description(movie: Movie) -> str:
    return f"Título: {movie.title_es}. Género: {movie.genre_tags}. Director: {movie.director_top_5}. Sinopsis: {movie.synopsis}"


#por ultimo utilizaremos un enfoque más minimalista.

def get_genre_and_title(movie: Movie) -> str:
    return f"Título: {movie.title_es}. Género: {movie.genre_tags}"