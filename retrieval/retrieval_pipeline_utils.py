import re
from nltk.corpus import stopwords

def clean_query_txt(texto: str) -> str:
    texto = texto.replace("El usuario busca ", "").strip()
    texto = texto.replace("película", "").strip()
    # Convertir la primera palabra a minúsculas
    texto = texto.lower()
    return texto