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

def get_clean_synopsis_txt(movie: Movie) -> str:
    movie_clean = movie.synopsis.replace("(FILMAFINITY)", "")
    
    return movie_clean

def get_clean_synopsis_txt_v2(movie: Movie) -> str:
    movie_clean = movie.synopsis.replace("(FILMAFINITY)", "")
    movie_clean = movie_clean + movie.genre_tags
    
    return movie_clean

def get_clean_synopsis_txt_v3(movie: Movie) -> str:
    movie_clean = movie.synopsis.replace("(FILMAFINITY)", "")
    genre_clean = movie.genre_tags.replace(";", ",")
    genre_clean = genre_clean.replace("&", ",")
    movie_clean = movie_clean + genre_clean
    
    return movie_clean