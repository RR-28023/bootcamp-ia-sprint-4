from __future__ import annotations

from langchain_core.documents import Document
import re

from data_utils import Movie
from retrieval import config


def create_docs_to_embedd(
    movies: list[Movie], config: config.RetrievalExpsConfig
) -> list[Document]:
    """
    Convierte una lista de objetos `Movie` a una lista the objetos `Document`
    (usada por Langchain). En esta función se decide que parte de los datos
    será usado como embeddings y que parte como metadata.
    """
    movies_as_docs = []
    for movie in movies:
        content = config.text_to_embed_fn(movie)
        metadata = movie.model_dump()
        doc = Document(page_content=content, metadata=metadata)
        movies_as_docs.append(doc)

    return movies_as_docs

def get_synopsys_txt_2(movie: Movie,) -> str:
    # Pre-limpieza
    synopsis_cleaned = re.sub(r'\(filmaffinity\)', '', movie.synopsis.lower())    
    # Preparar el texto completo
    pre_cleaned = f"{movie.title_es}. {synopsis_cleaned} {movie.genre_tags}"
    clean_movie = re.sub(r';', ',', pre_cleaned.lower())
    return clean_movie
