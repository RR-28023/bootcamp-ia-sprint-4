import nltk
import stanza 

stanza.download("es")

nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')

def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    return query

def modified_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    doc = nlp(query)
    lemas = [word.lemma  for sent in doc.sentences for word in sent.words]
    clean_query = " ".join(lemas)
    return clean_query