import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# Descargar recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializar recursos
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

def correct_spelling(query: str) -> str:
    corrected_query = []
    for word in query.split():
        corrected_word = spell.correction(word)
        if corrected_word is not None:  # Verificar si la corrección no es None
            corrected_query.append(corrected_word)
        else:
            corrected_query.append(word)  # Si no se puede corregir, mantener la palabra original
    return ' '.join(corrected_query)

def remove_stopwords(query: str) -> str:
    filtered_words = [word for word in query.split() if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize(query: str) -> str:
    lemmatized_words = [lemmatizer.lemmatize(word) for word in query.split()]
    return ' '.join(lemmatized_words)

def clean_query_txt(query: str) -> str:
    # Preprocesamiento avanzado de la consulta
    query = query.lower()
    query = remove_stopwords(query)
    query = lemmatize(query)
    return query
