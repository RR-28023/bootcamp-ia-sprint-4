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

def get_clean_synopsys(movie: Movie) -> str:
    synopsis = movie.synopsis
    title = movie.title_es
    genre = movie.genre_tags

    clean_synopsis = "La película se titula " + title + ", la película es de los géneros: " + genre + " y su sinopsis es: " + synopsis

    return clean_synopsis

def get_clean_synopsys_2(movie: Movie) -> str:
    synopsis = movie.synopsis
    title = movie.title_es
    country = movie.country
    genre = movie.genre_tags
    genre = genre.replace("&", "y")
    genre = genre.replace(";", " ")

    clean_synopsis = "La película se titula " + title + ", es de los géneros: " + genre + ", ha sido creada por " + country + " y su sinopsis es: " + synopsis

    return clean_synopsis
