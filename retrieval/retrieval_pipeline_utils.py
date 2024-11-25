def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

#vamos a hacer una mejor limpieza de datos.

def enhanced_clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    # Convertir a minúsculas para estandarizar
    query = query.lower()
    # Eliminar caracteres especiales para simplificar la query
    query = ''.join(char for char in query if char.isalnum() or char.isspace())
    return query