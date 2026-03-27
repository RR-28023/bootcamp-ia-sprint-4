from __future__ import annotations

from langchain_core.documents import Document

from data_utils import Movie
from retrieval import config
import re
import string


def create_docs_to_embedd(movies: list[Movie], config: config.RetrievalExpsConfig) -> list[Document]:
    """
    Convierte una lista de objetos `Movie` a una lista the objetos `Document`(usada por Langchain).
    En esta función se decide que parte de los datos será usado como embeddings y que parte como metadata.
    """
    movies_as_docs = []
    for movie in movies:
        content = config.text_to_embed_fn(movie)
        metadata = movie.model_dump()
        doc = Document(page_content=content, metadata=metadata)
        movies_as_docs.append(doc)

    return movies_as_docs


## Posibles funciones para usar como `text_to_embed_fn` en `RetrievalExpsConfig` ##


def get_synopsys_txt_3(movie: Movie) -> str:
    resultado=movie.synopsis+movie.genre_tags
    #print(resultado)
    return resultado

def get_synopsys_txt_4(movie: Movie) -> str:
    resultado=movie.synopsis+movie.genre_tags+movie.title_es
    #print(resultado)
    return resultado

def get_synopsys_txt_5(movie: Movie) -> str:
    resultado=movie.synopsis+" "+movie.genre_tags+" "+movie.title_es
    resultado=re.sub("FILMAFFINITY","",resultado)
    resultado=re.sub(r"\(","",resultado)
    resultado=re.sub(r"\)","",resultado)

    STOPWORDS = set([
        'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 
        'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 
        'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 
        'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 
        'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 
        'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 
        'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 
        'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 
        'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 
        'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 
        'tus', 'ellas', 'nosotras', 'vosotros', 'vosotras', 'os', 'mío', 
        'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 
        'suya', 'suyos', 'suyas', 'nuestro', 'nuestra', 'nuestros', 
        'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'esos', 
        'esas', 'estoy', 'estás', 'está', 'estamos', 'estáis', 'están'
    ])

    # Signos de puntuación
    resultado=resultado.translate(str.maketrans('','',string.punctuation))

    # Stop words
    for stopword in STOPWORDS:
        resultado=re.sub(rf'\b{stopword}\b','',resultado,flags=re.IGNORECASE)

    # Eliminar espacios seguidos
    resultado=re.sub(r'\s+',' ',resultado).strip()

    #print(resultado)
    return resultado

# def ...
