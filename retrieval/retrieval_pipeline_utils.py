import re
from nltk.corpus import stopwords

def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    # Eliminar múltiples espacios intermedios
    query = re.sub(r'\s+', ' ', query)
    # Convertir a minúsculas da peores resultados
    # Eliminar caracteres especiales, mantener solo letras y números
    query = re.sub(r'[^\w\s]', '', query)
    # Eliminar stopwords da peores resultados
    return query
