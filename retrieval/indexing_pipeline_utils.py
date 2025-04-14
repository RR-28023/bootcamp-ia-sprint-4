from __future__ import annotations

from langchain_core.documents import Document

from data_utils import Movie
from retrieval import config


def create_docs_to_embedd(movies: list[Movie], config: config.RetrievalExpsConfig) -> list[Document]:
  
    movies_as_docs = []
    for movie in movies:
        content = config.text_to_embed_fn(movie)
        metadata = movie.model_dump()
        doc = Document(page_content=content, metadata=metadata)
        movies_as_docs.append(doc)

    return movies_as_docs


## Posibles funciones para usar como `text_to_embed_fn` en `RetrievalExpsConfig` ##

#funcion que se llama en config.py

def getMovieData(movie: dict, embedding_model=None) -> dict:
   
    textual_data = f"{movie['synopsis']}; {' '.join(movie['genre_tags'].split(';'))}"
    
    movie_embedding = None
    if embedding_model:
        movie_embedding = embedding_model.predict([textual_data])[0] 
    
    return {
        "movie_id": movie["movie_id"],
        "title": movie["title_es"],
        "year": movie["year"],
        "genres": movie["genre_tags"].split(';'),
        "synopsis": movie["synopsis"],
        "embedding": movie_embedding
    }
