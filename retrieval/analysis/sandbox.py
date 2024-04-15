# %%
import sys
from dotenv import load_dotenv
import os
import logging
load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))
# Set current working directory to the root of the project
os.chdir(Path(__file__).parents[2])
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
idx_ejemplo = 54
query = eval_queries[idx_ejemplo]
query, expected_movie_id = query["query"], query["movie_id"]
retrieved_docs, t_elapsed = retrieval_pipeline(query, index, exp_config, logger)

# %%
expected_movie_doc = index.docstore.search(expected_movie_id)
expected_movie_doc.metadata
# %%
query
# %%
retrieved_docs[0].metadata
# %%
