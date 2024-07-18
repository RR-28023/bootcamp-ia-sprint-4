import json
import time
from datetime import datetime
import logging
import os

from langchain_core.documents import Document

from data_utils import Movie
from retrieval.config import RetrievalExpsConfig
from langchain_community.vectorstores import FAISS



def preprocesado():
    querie_input = input("Por favor, describa el tipo de película que está buscando: ")
    query=str(querie_input)
    return query

def retrieval(
    query: str,
    index: FAISS,
    config: RetrievalExpsConfig,
    logger: logging.Logger,
    verbose: bool = False,
) -> tuple[list[Document], float]:
    """
    Pipeline para recuperar documentos similares a la query dada. Devuelve los documentos, y el tiempo empleado
    """
    if verbose:
        logger.info(f"Recuperando documentos similares a la query {query}...")
    t = time.time()
    retrieved_docs = index.similarity_search(query, k=10)
    t_elapsed = time.time() - t
    if verbose:
        logger.debug(f"{len(retrieved_docs)} recuperados en {t_elapsed:.0f} segundos.")
    return retrieved_docs, t_elapsed

def postprocesado(retrieved_data_for_user: dict, satisfaccion: int, query: str, id_sesion: str):
    sesiones = {}
    sesiones["query"]= query
    sesiones["retrieved_data"]=retrieved_data_for_user
    sesiones["satisfaccion"]= satisfaccion
    file_path = "./retrieval/data/sesiones.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}
    else:
        data={}

    data[id_sesion] = sesiones

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)