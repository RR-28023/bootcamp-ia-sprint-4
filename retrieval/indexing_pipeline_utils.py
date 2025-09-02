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

# def ...

def get_genre_director_and_synopsis(movie: Movie) -> str: # genero, director y sipnosis
    genres = movie.genre_tags.replace(";", ", ") if movie.genre_tags else "Unknown genres"
    director = movie.director_top_5 or "Unknown director"
    synopsis = movie.synopsis or "No synopsis available."
    return f"Genres: {genres}. Directed by {director}. Synopsis: {synopsis}"

def get_full_metadata(movie: Movie) -> str: # descripcion completa
    title = movie.title_es or "Unknown Title"
    original_title = movie.title_original or "Unknown Original Title"
    year = movie.year or "Unknown Year"
    genres = movie.genre_tags.replace(";", ", ") if movie.genre_tags else "Unknown genres"
    director = movie.director_top_5 or "Unknown director"
    synopsis = movie.synopsis or "No synopsis available."
    return (f"Title: {title} ({original_title}). Year: {year}. Genres: {genres}. "
            f"Directed by {director}. Synopsis: {synopsis}")


def get_synopsis_and_genres(movie: Movie) -> str: #Combina sipnosis y genero y utiliza los valores predeterminados
    synopsis = movie.synopsis or "No synopsis available."
    genres = movie.genre_tags.replace(";", ", ") if movie.genre_tags else "Unknown genres"
    return f"Synopsis: {synopsis} Genres: {genres}"