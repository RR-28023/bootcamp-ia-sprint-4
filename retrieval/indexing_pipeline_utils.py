
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
def get_info_txt(movie: Movie) -> str:

    info:str = "\n".join([movie.title_es, movie.synopsis])
    return info

def get_more_info_txt(movie: Movie) -> str:

    info:str = "\n".join([movie.title_es, movie.synopsis, movie.genre_tags])
    return info

def get_more_more_info_txt(movie: Movie) -> str:

    info:str = "\n".join([movie.title_es, movie.country, movie.synopsis, movie.genre_tags])
    return info

# ## Posibles funciones para usar como `text_to_embed_fn` en `RetrievalExpsConfig` ##
# def get_info_txt(movie: Movie) -> str:

#     info:str = "\n".join([movie.title_es, movie.synopsis])
#     return info
