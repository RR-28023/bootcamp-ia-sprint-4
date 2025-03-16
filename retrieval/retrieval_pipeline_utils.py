import re
import string

def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()

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
    query=query.translate(str.maketrans('','',string.punctuation))

    # Stop words
    for stopword in STOPWORDS:
        query=re.sub(rf'\b{stopword}\b','',query,flags=re.IGNORECASE)

    # Eliminar espacios seguidos
    query=re.sub(r'\s+',' ',query).strip()

    #print(query)

    return query
