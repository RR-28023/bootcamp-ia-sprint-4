from __future__ import annotations
from langchain_core.documents import Document
from transformers import ElectraModel, ElectraTokenizer
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

## Embeddings de electra (aunque el resto se definió en config.py)
def get_electra_embeddings(movie: Movie, model: ElectraModel, tokenizer: ElectraTokenizer) -> str:
    text = movie.synopsis
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Puedes ajustar el tipo de pooling aquí
    return embeddings.tolist()

