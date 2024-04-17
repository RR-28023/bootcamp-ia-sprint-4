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

def get_enriched_movie_text(movie: Movie) -> str:
    """
    Combina varios atributos de la película para formar un texto enriquecido.
    Argumentos:
        movie (Movie): Un objeto película que contiene múltiples atributos descriptivos.
    Retorna:
        str: Un texto enriquecido que representa la película.
    """
    # Concatenar información relevante de la película
    attributes = [
        movie.title_es,
        movie.title_original,
        movie.director_top_5,
        movie.cast_top_5,
        movie.genre_tags,
        movie.synopsis
    ]
    enriched_text = ' '.join(attributes)
    return enriched_text

def get_synopsys_txt(movie: Movie) -> str:
    return movie.synopsis
