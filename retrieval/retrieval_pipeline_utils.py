def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

# retrieval/retrieval_pipeline_utils.py

def preprocess_query_no_el_usuario(query: str) -> str:
    return query.replace("El usuario busca", "").strip()

def preprocess_query_lowercase(query: str) -> str:
    return query.replace("El usuario busca", "").strip().lower()

def advanced_clean_query_txt(query: str) -> str:
    import re
    from unidecode import unidecode

    # Remover la frase inicial
    query = query.replace("El usuario busca ", "").strip()
    # Remover caracteres especiales
    query = re.sub(r"[^a-zA-Z0-9\s]", "", query)
    # Convertir a minúsculas
    query = query.lower()
    # Remover acentos
    query = unidecode(query)
    return query