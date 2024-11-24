from langchain_core.documents import Document
from data_utils import Movie


def get_enriched_text(movie: Movie) -> str:
    """
    Genera un texto enriquecido a partir de los datos de la película.
    """
    enriched_text = (
        f"Título: {movie.title_es if movie.title_es else movie.title_original}. "
        f"Sinopsis: {movie.synopsis}. "
        f"Géneros: {movie.genre_tags}. "
        f"Director: {movie.director_top_5}. "
        f"Año: {movie.year}. "
        f"País: {movie.country}. "
    )
    return enriched_text.strip()


def create_docs_to_embedd(movies: list[Movie], config) -> list[Document]:
    """
    Convierte una lista de objetos `Movie` a una lista de objetos `Document` (usada por Langchain).
    """
    movies_as_docs = []
    for movie in movies:
        print(f"Procesando película: ID={movie.movie_id}, title_es={movie.title_es}, title_original={movie.title_original}")
        content = get_enriched_text(movie)  # Texto enriquecido para embeddings

        metadata = {
            "movie_id": movie.movie_id,
            "title_es": movie.title_es,
            "title_original": movie.title_original,
            "genre_tags": movie.genre_tags,
            "synopsis": movie.synopsis
        }

        print(f"Metadata generado: {metadata}")  # Depuración

        movies_as_docs.append(Document(page_content=content, metadata=metadata))
    return movies_as_docs


