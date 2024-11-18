import re

def clean_query_txt(query: str) -> str:
    """
    Limpia únicamente el ruido más evidente de las consultas.
    """
    # Elimina prefijos innecesarios
    query = query.replace("El usuario busca ", "").strip()
    
    # Normaliza espacios
    query = " ".join(query.split())
    
    return query
