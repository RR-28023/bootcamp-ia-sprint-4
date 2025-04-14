# %%
import sys
import os
import logging
from pathlib import Path
import mlflow #prueba
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
# Set current working directory to the root of the project
os.chdir(root_dir)
from retrieval.main import *
CACHE_PATH = Path(__file__).parents[1] / ".cache"
# %%
# Generamos los embeddings
movies = get_movies_data()
exp_config = RetrievalExpsConfig()
logger = logging.getLogger("retrieval")
if not (CACHE_PATH / f"faiss_{exp_config.index_config_unique_id}").exists():
    t_elapsed = generate_index_pipeline(exp_config, logger)

# Los cargamos
embedder = load_embedder(exp_config)
embedder.show_progress = False
index = FAISS.load_local(
    CACHE_PATH / f"faiss_{exp_config.index_config_unique_id}",
    embeddings=embedder,
    allow_dangerous_deserialization=True,
)
eval_queries = load_eval_queries()

#%%
# Veamos qué tal funciona
idx_ejemplo = 204 #  16: 'JFK: Caso revisado'; 180: 'Un verano en Ibiza'; 233: 'La muerte de Stalin', 204: 'Anatomía de una caída',
query = eval_queries[idx_ejemplo]
query, expected_movie_id = query["query"], query["movie_id"]
retrieved_docs, t_elapsed = retrieval_pipeline(query, index, exp_config, logger)
expected_movie_doc = index.docstore.search(expected_movie_id)
expected_movie_doc.metadata

print(f"--- Query ---\n{query}\n")
print(f"--- Movie that we expect to retrieve ---") 
print("\n".join([f" · {k}: {v}" for k,v in expected_movie_doc.metadata.items()]))
print(f"\n--- Retrieved movies ---")
for i, doc in enumerate(retrieved_docs):
    print(f"MOVIE {i+1}")
    print("\n".join([f" · {k}: {v}" for k,v in doc.metadata.items()]))

#Prueba
import mlflow

def run_experiment(config: RetrievalExpsConfig, movies: list[Movie], eval_queries: list):
    """
    Ejecuta un experimento de recuperación y registra los resultados con MLFlow.
    """
    # Inicia una nueva ejecución en MLFlow
    with mlflow.start_run():
        # Registrar parámetros del experimento
        mlflow.log_params(config.exp_params)

        # Generar el índice
        logger = logging.getLogger("retrieval")
        index_path = CACHE_PATH / f"faiss_{config.index_config_unique_id}"
        if not index_path.exists():
            generate_index_pipeline(config, logger)

        # Cargar el índice
        embedder = load_embedder(config)
        embedder.show_progress = False
        index = FAISS.load_local(
            index_path,
            embeddings=embedder,
            allow_dangerous_deserialization=True,
        )

        # Evaluar las queries
        total_score = 0
        for query_data in eval_queries:
            query, expected_movie_id = query_data["query"], query_data["movie_id"]
            retrieved_docs, _ = retrieval_pipeline(query, index, config, logger)
            if any(doc.metadata["movie_id"] == expected_movie_id for doc in retrieved_docs):
                total_score += 1

        # Calcular y registrar métricas
        retrieval_score = total_score / len(eval_queries)
        mlflow.log_metric("retrieval_score", retrieval_score)
