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
    response = dict()
    response["title_es"] = movie.title_es 
    response["title_original"] = movie.title_original
    response["duration_mins"] = movie.duration_mins
    response["year"] = movie.year
    response["genre_tags"] = movie.genre_tags
    response["director_top_5"] = movie.director_top_5
    response["script_top_5"] = movie.script_top_5
    response["cast_top_5"] = movie.cast_top_5
    response["photography_top_5"] = movie.photography_top_5
    response["synopsis"] = movie.synopsis
    print(response)
    return str(response)

# def ...
