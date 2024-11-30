import logging
import os
import time
from pathlib import Path
import sys
import requests

# Append the workspace's root directory to the sys.path
sys.path.append(str(Path(__file__).parent.parent))

import colorlog
import mlflow
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from data_utils import get_movies_data
from retrieval.config import RetrievalExpsConfig
from retrieval.indexing_pipeline_utils import create_docs_to_embedd

CACHE_PATH = Path(__file__).parent / ".cache"


# Fase 1: Preprocesado
def preprocess_query(raw_query: str, config: RetrievalExpsConfig, embedder) -> dict:
    return config.query_prepro_fn(raw_query, embedding_model=embedder)


# Fase 2: Retrieval
def retrieve_documents(preprocessed_query: str, index: FAISS, k: int = 10) -> list[Document]:
    return index.similarity_search(preprocessed_query, k=k)


# Fase 3: Posprocesado
def postprocess_results(retrieved_docs: list[Document]) -> list[dict]:
    return [
        {
            "movie_id": doc.metadata["movie_id"],
            "title": doc.metadata.get("title_es", ""),
            "synopsis": doc.page_content,
            "genres": doc.metadata.get("genre_tags", ""),
        }
        for doc in retrieved_docs
    ]

#ESTAS 3 FUNCIONES SUSTITUYEN LA FUNCION ANTERIOR RETRIEVAL_PIPELINE

def load_embedder(config: RetrievalExpsConfig) -> HuggingFaceEmbeddings:
    encode_kwargs = {"normalize_embeddings": config.normalize_embeddings}
    embedder = HuggingFaceEmbeddings(
        model_name=config.model_name,
        multi_process=False,
        show_progress=True,
        encode_kwargs=encode_kwargs,
    )
    return embedder


def generate_index_pipeline(config: RetrievalExpsConfig, logger: logging.Logger) -> float:
    logger.info("Cargando los datos de las películas...")
    movies = get_movies_data()
    logger.info(f"Se han cargado datos sobre {len(movies):,} películas.")

    logger.info("Convirtiendo los datos en Documentos de Langchain...")
    movies_as_docs: list[Document] = create_docs_to_embedd(movies, config)

    logger.info("Generando los embeddings y creando el índice...")
    t0 = time.time()
    embedder = load_embedder(config)
    movie_ids = [doc.metadata["movie_id"] for doc in movies_as_docs]
    index = FAISS.from_documents(movies_as_docs, embedder, ids=movie_ids)

    # Guardamos el índice en local
    path_to_save_index = CACHE_PATH / f"faiss_{config.index_config_unique_id}"
    path_to_save_index.mkdir(parents=True, exist_ok=True)
    index.save_local(path_to_save_index)
    t_elapsed = time.time() - t0
    logger.info(f"Índice generado y guardado en {t_elapsed:.0f} segundos.")
    return t_elapsed


if __name__ == "__main__":

    # Configuración del logger
    logger = colorlog.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(asctime)s-%(name)s-%(log_color)s%(levelname)s%(reset)s: %(message)s", datefmt="%y/%m/%d-%H:%M:%S"
        )
    )
    logger.addHandler(handler)

    # Configuración de MLFlow
    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("Embeddings Retrieval")

    with mlflow.start_run():
        exp_config = RetrievalExpsConfig()
        mlflow.log_params(exp_config.exp_params)

        # Generar índice si no existe
        index_path = CACHE_PATH / f"faiss_{exp_config.index_config_unique_id}"
        if not index_path.exists():
            generate_index_pipeline(exp_config, logger)

        # Cargar el índice y el embedder
        embedder = load_embedder(exp_config)
        index = FAISS.load_local(
            index_path,
            embeddings=embedder,
            allow_dangerous_deserialization=True,
        )

        # Ejemplo de pipeline completo con una query
        raw_query = "El usuario busca una película de romance ambientada en los años 30"
        logger.info(f"Query original: {raw_query}")

        # Preprocesado
        preprocessed_query = preprocess_query(raw_query, exp_config, embedder)
        logger.info(f"Query procesada: {preprocessed_query['query_cleaned']}")

        # Retrieval
        retrieved_docs = retrieve_documents(preprocessed_query["query_cleaned"], index)
        logger.info(f"Documentos recuperados: {len(retrieved_docs)}")

        # Posprocesado
        results = postprocess_results(retrieved_docs)
        logger.info(f"Resultados estructurados: {results}")

        # Guardar resultados en MLFlow
        mlflow.log_text(str(results), "retrieved_results.json")
