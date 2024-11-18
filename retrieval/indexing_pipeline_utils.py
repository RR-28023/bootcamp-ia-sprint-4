from __future__ import annotations

from langchain_core.documents import Document
from data_utils import Movie
from retrieval import config


def create_docs_to_embedd(movies: list[Movie], config: config.RetrievalExpsConfig) -> list[Document]:
    """
    Convierte una lista de objetos `Movie` a una lista de objetos `Document` (usada por Langchain).
    """
    movies_as_docs = []
    for movie in movies:
        content = config.text_to_embed_fn(movie)
        metadata = movie.model_dump()
        doc = Document(page_content=content, metadata=metadata)
        movies_as_docs.append(doc)

    return movies_as_docs


def get_synopsys_txt(movie: Movie) -> str:
    """
    Devuelve el texto de la sinopsis para los embeddings.
    """
    return movie.synopsis


def get_enriched_text(movie: Movie) -> str:
    """
    Genera un texto enriquecido a partir de los datos de la película.
    """
    enriched_text = (
        f"Título: {movie.title_es if movie.title_es else movie.title_original}. "
        f"Sinopsis: {movie.synopsis}. "
        f"Géneros: {movie.genre_tags}. "
        f"Director: {movie.director_top_5}. "
        f"Año: {movie.year}. "
        f"País: {movie.country}. "
    )
    return enriched_text.strip()
