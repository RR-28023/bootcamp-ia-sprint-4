# %%
import sys
from pathlib import Path
# Sort out paths for proper imports
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
os.chdir(root_dir)
from retrieval.main import *
import numpy as np

# %%
# Load already generated embeddings from the cache
movies = get_movies_data()
exp_config = RetrievalExpsConfig()
embedder = load_embedder(exp_config)
index = FAISS.load_local(
    CACHE_PATH / f"faiss_{exp_config.index_config_unique_id}",
    embeddings=embedder,
    allow_dangerous_deserialization=True,
)
logger = logging.getLogger(__name__)
# %%
movie = [m for m in movies if m.title_es == "Ex Machina"][0]
print(movie.synopsis)
print(movie.url)
# Get top 10 most similar synopsis
query = movie.synopsis
retrieved_docs, t_elapsed = retrieval_pipeline(query, index, exp_config, logger)
most_similar_movie_id = retrieved_docs[1].metadata["movie_id"]
most_similar_movie = [m for m in movies if m.movie_id == most_similar_movie_id][0]
print(most_similar_movie.title_es)
print(most_similar_movie.synopsis)
print(most_similar_movie.url)

# %%
genres = [g for m in movies for g in m.genre_tags.split(";")]
unique_genres = set(genres)
unique_genres

# Embeddings
genre_embeddings = embedder.embed_documents(list(unique_genres))
genre_embeddings = np.array(genre_embeddings)

# Reduce dimensionality
# %%
import umap
import matplotlib.pyplot as plt
reducer = umap.UMAP(n_components=2, random_state=42)
genre_embeddings_2d = reducer.fit_transform(genre_embeddings)

# %%
#Create scatter plot
plt.figure(figsize=(15, 10))
plt.scatter(genre_embeddings_2d[:, 0], genre_embeddings_2d[:, 1], alpha=0.5)

# Add labels for a subset of points (e.g., 30% of points)
n_labels = int(len(unique_genres) * 0.3)
indices = np.random.choice(len(unique_genres), n_labels, replace=False)

for idx in indices:
    genre = list(unique_genres)[idx]
    x, y = genre_embeddings_2d[idx]
    plt.annotate(genre, (x, y), xytext=(5, 5), textcoords='offset points')

plt.title('Genre Embeddings Visualization (2D)')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()
plt.show()
# %%
