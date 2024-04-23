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

def get_sy_au_dir(movie: Movie) -> str:
    # Construir la parte de la sinopsis
    str_mov = movie.synopsis + ", "

    # Agregar los autores del guion
    str_mov += "Autores: " + movie.script_top_5 + ", "

    # Agregar el director
    str_mov += "Director: " + movie.director_top_5

    return str_mov
def get_sy_au_dir_year(movie: Movie) -> str:
    # Construir la parte de la sinopsis
    str_mov = movie.synopsis + ", "

    # Agregar los autores del guion
    str_mov += "Autores: " + movie.script_top_5 + ", "

    # Agregar el director
    str_mov += "Director: " + movie.director_top_5 + ","
    str_mov += "Año: " + movie.year
    
    return str_mov

# def ...
