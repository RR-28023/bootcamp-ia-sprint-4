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
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Función para extraer información relevante como un string
def get_synopsys_txt(movie: Movie) -> str:
    # Limpiar la sinopsis eliminando caracteres especiales y dejando texto relevante
    synopsis_cleaned = re.sub(r'\s+', ' ', movie.synopsis.strip())  # Remueve espacios múltiples 
    movie.genre_tags.lower()

    # Generar texto enriquecido para embeddings
    enriched_text = (
        f"{movie.genre_tags.replace(";", " ").strip()}. {synopsis_cleaned}"
    )
    print(enriched_text)
    return enriched_text