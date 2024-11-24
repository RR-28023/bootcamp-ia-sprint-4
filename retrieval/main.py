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
from tqdm import tqdm

from data_utils import get_movies_data
from retrieval.config import RetrievalExpsConfig
from retrieval.evaluation import (
    calc_mrr,
    comparar_resultado_con_esperado,
    is_in_results,
    load_eval_queries,
    plot_rank_distribution,
)
from retrieval.indexing_pipeline_utils import create_docs_to_embedd
from retrieval_pipeline_utils import preprocesado, posprocesado

CACHE_PATH = Path(__file__).parent / ".cache"


def load_embedder(config: RetrievalExpsConfig) -> HuggingFaceEmbeddings:
    """
    Carga el modelo de embeddings configurado.
    """
    encode_kwargs = {"normalize_embeddings": config.normalize_embeddings}
    embedder = HuggingFaceEmbeddings(
        model_name=config.model_name,
        multi_process=False,
        show_progress=True,
        encode_kwargs=encode_kwargs,
    )
    return embedder


def generate_index_pipeline(config: RetrievalExpsConfig, logger: logging.Logger) -> float:
    """
    Pipeline para generar y guardar el índice con los detalles de las películas. Devuelve el tiempo que ha tardado en
    generar el índice (p.ej. para logearlo en mlflow)
    """
    logger.info("Cargando los datos de las películas...")
    movies = get_movies_data()
    logger.info(f"Se han cargado datos sobre {len(movies):,} películas.")

    logger.info("Convirtiendo los datos en Documentos de Langchain...")
    movies_as_docs: list[Document] = create_docs_to_embedd(movies, config)

    logger.info("Cargando el modelo para generar retrieval..")

    logger.info("Generando los embeddings y creando el índice")
    t0 = time.time()
    embedder = load_embedder(config)
    movie_ids = [doc.metadata["movie_id"] for doc in movies_as_docs]
    index = FAISS.from_documents(movies_as_docs, embedder, ids=movie_ids)

    path_to_save_index = CACHE_PATH / f"faiss_{config.index_config_unique_id}"
    path_to_save_index.mkdir(parents=True, exist_ok=True)
    index.save_local(path_to_save_index)
    t_elapsed = time.time() - t0
    logger.info(f"Se han generado y guardado los embeddings en {t_elapsed:.0f} segundos.")
    return t_elapsed


def retrieval(query_txt: str, index_path: Path, config: RetrievalExpsConfig) -> list[Document]:
    """
    Recupera los documentos relevantes usando el índice y el modelo.
    """
    # Cargar el modelo desde MLFlow
    embedder = load_embedder(config)
    embedder.show_progress = False

    # Cargar el índice FAISS
    index = FAISS.load_local(
        index_path,
        embeddings=embedder,
        allow_dangerous_deserialization=True,
    )
    
    # Realizar la búsqueda
    retrieved_docs = index.similarity_search(query_txt, k=10)
    return retrieved_docs


def flujo_inferencia(query: str, index_path: Path, config: RetrievalExpsConfig) -> list[dict]:
    """
    Flujo completo de inferencia compuesto por preprocesado, retrieval y posprocesado.
    """
    # 1. Preprocesado
    query_txt = preprocesado(query, config)
    
    # 2. Retrieval
    retrieved_docs = retrieval(query_txt, index_path, config)
    
    # 3. Posprocesado
    results = posprocesado(retrieved_docs)
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    import logging

    # Configuración
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Crear configuración e inicializar preprocesador
    config = RetrievalExpsConfig()
    config.initialize_preprocessor()

    # Imprimir el nombre del índice esperado
    logger.info(f"Nombre del índice esperado: {config.index_config_unique_id}")

    # Ruta al índice
    index_path = CACHE_PATH / f"faiss_{config.index_config_unique_id}"

    # Comprobar si el índice existe
    if not index_path.exists():
        logger.error(f"No se encontró el índice esperado en la ruta: {index_path}")
    else:
        logger.info(f"Índice encontrado: {index_path}")

    # Si el índice existe, ejecuta la inferencia
    if index_path.exists():
        # Ejemplo de consulta
        query = "El usuario busca un drama histórico sobre la guerra, basado en hechos reales."

        # Ejecución del flujo de inferencia
        results = flujo_inferencia(query, index_path, config)

        # Mostrar resultados
        for result in results:
            print(result)

