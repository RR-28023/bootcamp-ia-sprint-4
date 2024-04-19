## Funciones para Experimentos de Recuperación

from data_utils.schemas import Movie
import re

def get_title_txt(movie: Movie) -> str:
    """
    Función para obtener el título de una película.
    """
    return movie.title

def clean_query_txt(query: str) -> str:
    """
    Función para limpiar la query de entrada.
    """
    # Convertir todas las letras a minúsculas
    query = query.lower()
    
    # Eliminar la frase "El usuario busca" si está presente
    query = re.sub(r'^el usuario busca', '', query)
    
    # Eliminar caracteres de puntuación
    query = re.sub(r'[^\w\s]', '', query)
    
    # Eliminar espacios adicionales
    query = query.strip()
    
    return query
