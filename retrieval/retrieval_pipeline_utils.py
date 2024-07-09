def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

# retrieval/retrieval_pipeline_utils.py

def preprocess_query_no_el_usuario(query: str) -> str:
    return query.replace("El usuario busca", "").strip()

def preprocess_query_lowercase(query: str) -> str:
    return query.replace("El usuario busca", "").strip().lower()
