def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

def clean_query_txt2(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    
    query = query.lower()
    
    stop_words = {'un', 'una', 'el', 'la', 'sobre', 'en', 'de', 'con', 'específicamente', 'que', 'se', 'y'}
    punctuations = '.,;:!¿?()[]{}"\''
    query = ''.join([char for char in query if char not in punctuations])
    
    words = query.split()
    keywords = [word for word in words if word not in stop_words]
    
    cleaned_query = ' '.join(keywords)
    return cleaned_query