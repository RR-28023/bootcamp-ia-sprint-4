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

# Metodos o funciones que se utilizan para obtener los datos de la bbdd en un objeto
# "Movie", y se preprocesan para utilizarlos en la query o prompt de input al modelo
# preprocesado de generative txt..[RGA]

def get_synopsys_txt(movie: Movie) -> str:
    return movie.synopsis

# def ...

def get_dataMovies_transform_txt(movie: Movie) -> str:
    
    synopsis = movie.synopsis.replace("(FILMAFINITY)", "")
    genres = movie.genre_tags.lower()
    data_movie = synopsis + genres

    return data_movie
