# Metodos o funciones que procesan la query original y la transforman
# para generar un "prompt" más entendible para los modelos preentrenados..[RGA]  

def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

def clean_query_txt_v2(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query