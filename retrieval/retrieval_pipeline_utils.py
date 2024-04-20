import re

def clean_query_txt(query: str) -> str:
    if query.startswith("El usuario busca "):
        query = query[len("El usuario busca "):]  # Eliminar la parte inicial de la cadena
    query = query.strip()  # Eliminar espacios en blanco al inicio y al final de la cadena
    # Eliminar signos de puntuación
    query = re.sub(r'[^\w\s]', '', query)
    # Convertir a minúsculas
    query = query.lower()
    # Eliminar palabras comunes
    palabras_comunes = ['un', 'una', 'sobre', 'en', 'de', 'que', 'con', 'el', 'la', 'los', 'las', 'y']
    query = ' '.join([palabra for palabra in query.split() if palabra not in palabras_comunes])
    return query
