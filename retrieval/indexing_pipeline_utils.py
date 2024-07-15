from __future__ import annotations

from langchain_core.documents import Document

from data_utils import Movie
from retrieval import config
import torch

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


def get_synopsys_txt(movie: Movie) -> str:
    return movie.synopsis

#def get_synopsys_txt(movie: Movie) -> str:
    # Obtener la sinopsis de la película
    #text = movie.synopsis
    
    # Utilizar el tokenizer y el modelo desde config.py
    #inputs = config.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Obtención de las salidas del modelo BERT
    #with torch.no_grad():
    #    outputs = config.model(**inputs)
    
    # La salida del modelo BERT contiene varias representaciones, tomaremos la primera (hidden state)
    #embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    # Convertir el embedding a una lista para que sea serializable
    #embeddings_list = embeddings.tolist()
    
    #return embeddings_list
