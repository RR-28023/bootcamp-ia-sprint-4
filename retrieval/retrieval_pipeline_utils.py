import re
#from unidecode import unidecode

def clean_query_txt_caracteres_especiales(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    query = re.sub(r'[^\w\s]', '', query)
    query = query.lower()
    #query = unidecode(query)
    return query
