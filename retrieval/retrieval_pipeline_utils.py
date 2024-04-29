n=0
def clean_query_txt(query: str) -> str:
    global n
    query = query.replace("El usuario busca ", "").strip()
    query = eliminar_palabras_cortas(query)

    if n%100==0:
        n=0
        print(query)
    n +=1
    return query
import re

def eliminar_palabras_cortas(texto):
    # Define la expresión regular para encontrar palabras de longitud <= 3 en español
    patron = r'\b\w{1}\b'  # \b indica límites de palabras, \w{1,3} encuentra palabras de longitud 1 a 3

    # Sustituye las palabras encontradas por una cadena vacía
    texto_procesado = re.sub(patron, '', texto)

    return texto_procesado

