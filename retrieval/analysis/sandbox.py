# %%
import sys
import os
import logging
from pathlib import Path
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


# %%
