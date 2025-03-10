import re
from langchain_core.documents import Document

def clean_query_txt(query: str) -> str:
    """
    Limpia únicamente el ruido más evidente de las consultas.
    """
    query = query.replace("El usuario busca ", "").strip()
    query = " ".join(query.split())
    return query

def preprocesado(query: str, config) -> str:
    """
    Limpia y preprocesa la consulta del usuario.
    """
    from retrieval.config import RetrievalExpsConfig  # Importación local
    if isinstance(config, RetrievalExpsConfig):
        return config.query_prepro_fn(query)
    raise ValueError("El parámetro 'config' no es una instancia de RetrievalExpsConfig")

def posprocesado(retrieved_docs: list[Document]) -> list[dict]:
    """
    Convierte los documentos recuperados en una lista de resultados estructurados.
    """
    results = []
    for rank, doc in enumerate(retrieved_docs, start=1):
        metadata = doc.metadata
        results.append({
            "movie_id": metadata.get("movie_id", "N/A"),
            "title_es": metadata.get("title_es", "Sin título en español"),
            "title_original": metadata.get("title_original", "Sin título original"),
            "genre": metadata.get("genre_tags", "Desconocido"),
            "synopsis": metadata.get("synopsis", "Sin sinopsis disponible"),
            "rank": rank
        })
    return results



