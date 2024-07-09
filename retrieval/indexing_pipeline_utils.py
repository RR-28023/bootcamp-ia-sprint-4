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

def get_title_and_genre_txt(movie: Movie) -> str:
    return f"Título: {movie.title_es}. Géneros: {movie.genre_tags}."

def get_director_and_cast_txt(movie: Movie) -> str:
    return f"Director: {movie.director_top_5}. Elenco: {movie.cast_top_5}."

def combined_text(movie: Movie) -> str:
    """
    Combina varios atributos de la película para generar un texto más completo.
    """
    attributes = [
        movie.title_es,
        movie.title_original,
        f"Duración: {movie.duration_mins} minutos",
        f"Año: {movie.year}",
        f"País: {movie.country}",
        f"Géneros: {movie.genre_tags}",
        f"Director: {movie.director_top_5}",
        f"Guionistas: {movie.script_top_5}",
        f"Reparto: {movie.cast_top_5}",
        f"Fotografía: {movie.photography_top_5}",
        movie.synopsis,
    ]
    return " | ".join(attributes)

def get_detailed_txt(movie: Movie) -> str:
    return f"{movie.synopsis} Género: {movie.genre_tags}. Actores: {movie.cast_top_5}."