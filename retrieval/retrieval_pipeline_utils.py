import re

def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    query = query.lower()

    # Eliminar caracteres especiales, números, and signos de puntuación
    query = re.sub(r'[^a-záéíóúñ\s]', '', query)

    return query
