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

def get_movie_txt(movie: Movie) -> str:
    content_type_str = "serie" if movie.tv_show_flag else "película"
    genre_str = movie.genre_tags.replace(";", ", ").replace("/", " ").replace("&", "y")
    synopsis_str = movie.synopsis.replace("(FILMAFFINITY)", "")

    return f"Es una {content_type_str} de género {genre_str}, su país de origen es {movie.country}. Su argumento es el siguiente {synopsis_str}"
