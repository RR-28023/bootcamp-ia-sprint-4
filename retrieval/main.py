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

CACHE_PATH = Path(__file__).parent / ".cache"


def load_embedder(config: RetrievalExpsConfig) -> HuggingFaceEmbeddings:
    encode_kwargs = {"normalize_embeddings": config.normalize_embeddings}
    # Ver: https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html#langchain-community-embeddings-huggingface-huggingfaceembeddings
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
    # El número total de películas se ha capado a 6,521 para el ejercicio, así los tiempos de ejecución de los experimentos
    # son más razonables en un ordenador personal
    logger.info(f"Se han cargado datos sobre {len(movies):,} películas.")

    logger.info("Convirtiendo los datos en Documentos de Langchain...")
    # En este paso se decide qué parte de los datos será usada como embeddings y que parte como metadata
    movies_as_docs: list[Document] = create_docs_to_embedd(movies, config)

    logger.info("Cargando el modelo para generar retrieval..")

    logger.info("Generando los embeddings y creando el índice")
    t0 = time.time()
    embedder = load_embedder(config)
    movie_ids = [doc.metadata["movie_id"] for doc in movies_as_docs]
    index = FAISS.from_documents(movies_as_docs, embedder, ids=movie_ids)

    # Guardamos el índice en local
    path_to_save_index = CACHE_PATH / f"faiss_{exp_config.index_config_unique_id}"
    path_to_save_index.mkdir(parents=True, exist_ok=True)
    index.save_local(path_to_save_index)
    t_elapsed = time.time() - t0
    logger.info(f"Se han generado y guardado los embeddings en {t_elapsed:.0f} segundos.")
    return t_elapsed


def retrieval_pipeline(
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
    query_txt = config.query_prepro_fn(query)
    retrieved_docs = index.similarity_search(query_txt, k=10)
    t_elapsed = time.time() - t
    if verbose:
        logger.debug(f"{len(retrieved_docs)} recuperados en {t_elapsed:.0f} segundos.")
    return retrieved_docs, t_elapsed

def find_movie_by_id(movies, target_id):
    for movie in movies:
        if movie.movie_id == target_id:
            return movie
    return None

if __name__ == "__main__":

    # Configuramos el logging
    logger = colorlog.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(asctime)s-%(name)s-%(log_color)s%(levelname)s%(reset)s: %(message)s", datefmt="%y/%m/%d-%H:%M:%S"
        )
    )
    logger.addHandler(handler)

    # Configuramos mlflow
    mlflow.set_tracking_uri("http://localhost:8080")
    # Check that there is a running tracking server on the given URI
    try:
        response = requests.get("http://localhost:8080")
    except requests.exceptions.ConnectionError:
        logger.error("No se ha podido conectar con el servidor de mlflow. ¿Está arrancado?")
        sys.exit(1)

    mlflow.set_experiment("Embeddings Retrieval")

    with mlflow.start_run():

        # Cargamos la configuración del experimento y logeamos a mlflow
        exp_config = RetrievalExpsConfig()
        mlflow.log_params(exp_config.exp_params)

        # Generamos el índice (sólo si no se ha generado uno ya con la misma configuracion)
        if not (CACHE_PATH / f"faiss_{exp_config.index_config_unique_id}").exists():
            t_elapsed = generate_index_pipeline(exp_config, logger)
            # Save the metric in the cache
            with open(CACHE_PATH / f"time_elapsed_{exp_config.index_config_unique_id}.txt", "w") as f:
                f.write(str(t_elapsed))

        # Evaluamos el pipeline de generación de índice y el de retrieval
        eval_queries = load_eval_queries()
        logger.info(f"Evaluando el modelo de retieval con {len(eval_queries):,} queries...")
        logger.info(f"Cargando el índice con los embeddings..")
        embedder = load_embedder(exp_config)
        embedder.show_progress = False
        index = FAISS.load_local(
            CACHE_PATH / f"faiss_{exp_config.index_config_unique_id}",
            embeddings=embedder,
            allow_dangerous_deserialization=True,
        )
        index_gen_secs = float(open(CACHE_PATH / f"time_elapsed_{exp_config.index_config_unique_id}.txt").read())
        mlflow.log_metric("index_gen_minutes", round(index_gen_secs/60,1))

        # Comenzamos el loop de evaluación
        mean_mrr, perc_in_top_10, n, accum_time = 0.0, 0.0, 0, 0.0
        ranks = []
        for query in tqdm(eval_queries):
            query, expected_movie_id = query["query"], query["movie_id"]
            retrieved_docs, t_elapsed = retrieval_pipeline(query, index, exp_config, logger)
            accum_time += t_elapsed
            retrieved_movies_ids = [doc.metadata["movie_id"] for doc in retrieved_docs]
            mrr, rank = calc_mrr(expected_movie_id, retrieved_movies_ids)
            ranks.append(rank)
            in_top_10 = is_in_results(expected_movie_id, retrieved_movies_ids)
            n += 1
            mean_mrr = (mean_mrr * (n - 1) + mrr) / n
            perc_in_top_10 = (perc_in_top_10 * (n - 1) + in_top_10) / n
            if mrr == 0.0 and logger.level == logging.DEBUG:
                # Para debugear
                expected_movie_doc = index.docstore.search(expected_movie_id)
                debug_str = comparar_resultado_con_esperado(query, retrieved_docs[0], expected_movie_doc, exp_config)
                logger.debug(f"La peli buscada no estaba en las 10 recuperadas: {debug_str}")
            
            movies = get_movies_data()
            movie = find_movie_by_id(movies, retrieved_movies_ids[0])
            print('Pelicula mas recomendada')
            print(movie.title_original)
            print(movie.synopsis)

        logger.info(f"MRR@10: {mean_mrr:.3f}")
        logger.info(
            f"Porcentaje de queries en las que la peli buscada estaba en las 10 recuperadas: {perc_in_top_10:.1%}"
        )
        mlflow.log_metrics(
            {
                "mean_mrr10": round(mean_mrr, 3),
                "perc_top_10": round(perc_in_top_10 * 100, 1),
                "secs_per_query": round(accum_time / n, 2),
            }
        )
        ranks_fig = plot_rank_distribution(ranks)
        mlflow.log_figure(ranks_fig, "rank_distribution.png")