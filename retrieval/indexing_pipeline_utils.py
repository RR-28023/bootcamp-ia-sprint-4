from __future__ import annotations

import re

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


def get_synopsys_and_genre_txt(movie: Movie) -> str:
    return f'synopsis: {movie.synopsis}, genre: {movie.genre_tags}'


def get_clean_synopsis_and_genre_txt(movie: Movie) -> str:
    file = open(
        "/Users/vladimirmaksimov/Dev/Todo/bootcamp-ia-sprint-4/retrieval/data/spanish.txt",
        encoding="utf8"
    )
    stop_words = []
    for stop_word in file:
        stop_words.append(str(stop_word.strip()))
    synopsis = movie.synopsis
    synopsis = synopsis.replace("(FILMAFFINITY)", "").strip()
    # Delete special characters and punctuation.
    synopsis = re.sub(r'[^A-Za-z\s, á, é, ú, ü, í, ó, ñ, 0-9]', '', synopsis)
    synopsis = re.sub(r'[,]+' , '' , synopsis)
    # Replace upper letters with lower ones.
    synopsis = synopsis.lower()
    # Split the query into the list of words.
    synopsis_list = list(synopsis.split())
    # Join the words into a phrase without stop words.
    synopsis = ' '.join([word for word in synopsis_list if word not in stop_words])
    return f'synopsis: {synopsis}, genre: {movie.genre_tags}'

