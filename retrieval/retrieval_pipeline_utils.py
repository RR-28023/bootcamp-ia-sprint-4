import re
def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

def clean_query_txt_2(query: str) -> str:
    query = query.lower()
    query = query.replace("el usuario busca ", "").strip()
    # Eliminar caracteres especiales y números utilizando expresiones regulares
    query = re.sub(r'[^a-zA-Z\s]', '', query)
    return query