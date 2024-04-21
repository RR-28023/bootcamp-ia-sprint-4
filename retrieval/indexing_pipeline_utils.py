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


def get_synopsys_txt(movie: Movie) -> str:
    return movie.synopsis


def get_synopsys_and_metadata_txt(movie: Movie) -> str:
    metadata = [
        movie.title_es,
        movie.title_original, 
        str(movie.year),
        movie.country,
        movie.genre_tags,
        movie.director_top_5,
        movie.cast_top_5
    ]
    return " ".join([movie.synopsis] + metadata)