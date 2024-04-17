def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query


def clean_query_txt_new(query: str) -> str:
    """
    Limpia la consulta eliminando frases comunes y espacios innecesarios,
    además de convertir todo el texto a minúsculas para estandarización.
    
    Args:
    query (str): La consulta original proporcionada por el usuario.

    Returns:
    str: La consulta limpia y normalizada.
    """
    # Definir frases comunes a eliminar
    common_prefixes = [
        "El usuario busca ",
        "Tema de la película: ",
        "Encuentra información sobre "
    ]
    
    # Convertir a minúsculas para homogeneizar la comparación
    query = query.lower()
    
    # Eliminar frases comunes
    for prefix in common_prefixes:
        if query.startswith(prefix):
            query = query[len(prefix):]
            break  # Suponemos que solo hay un prefijo para eliminar

    # Eliminar espacios adicionales y espacios al inicio y final
    query = query.strip()
    
    # Opcional: eliminar puntuación si se considera innecesaria
    # query = ''.join([char for char in query if char not in string.punctuation])

    return query