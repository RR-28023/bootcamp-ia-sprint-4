from __future__ import annotations

from typing import Callable

from data_utils import Movie
#añadir todas las funciones creadas
from retrieval.indexing_pipeline_utils import getMovieData
from retrieval.retrieval_pipeline_utils import mod_query

from transformers import AutoModel, AutoTokenizer
import numpy as np

class RetrievalExpsConfig:

    def __init__(self):

         # Definición del modelo
        # modelo empleado, se puede usar cualquiera de hugging face
        #self.model_name: str = "mrm8488/multilingual-e5-large-ft-sts-spanish-matryoshka-768-64-5e"
        self.model_name: str = "Shaharyar6/finetuned_sentence_similarity_spanish"
        # Función a emplear para generar el texto a indexar con embeddings; Debe tomar como input un objeto `Movie` y devolver un string
        self._text_to_embed_fn: Callable = getMovieData

        self.normalize_embeddings: bool = False  # Normalizar los embeddings a longitud 1 antes de indexarlos

        self._query_prepro_fn: Callable = mod_query

        self.embedding_model = self.load_embedding_model()


        # Parámetros para la generación de embeddings

    @staticmethod
    def normalize_embedding(embedding):
        """
        Normaliza un vector de embeddings a longitud 1.
        """
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def text_to_embed_fn(self, movie: Movie | dict) -> str:
   
        if isinstance(movie, Movie):
            return self._text_to_embed_fn(movie)
        elif isinstance(movie, dict):
        # Supongamos que getMovieData acepta diccionarios ahora
            return getMovieData(movie)
        else:
            raise ValueError("Formato de película no soportado. Use Movie o dict.")


    def query_prepro_fn(self, query: str, embedding_model) -> dict:
        return self._query_prepro_fn(query, embedding_model)


        # Método para cargar el modelo
    def load_embedding_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        return {"tokenizer": tokenizer, "model": model}

    @property
    def index_config_unique_id(self) -> str:
        mname = self.model_name.replace("/", "_")
        return f"{mname}_{self._text_to_embed_fn.__name__}_{self.normalize_embeddings}"

    @property
    def exp_params(self) -> dict:
        """
        Return the config parameters as a dictionary. To be used, for example, in mlflow logging
        """
        return {
            "model_name": self.model_name,
            "text_to_embed_fn": self._text_to_embed_fn.__name__,
            "normalize_embeddings": self.normalize_embeddings,
            "query_prepro_fn": self._query_prepro_fn.__name__,
            "embedding_model_loaded": hasattr(self, "embedding_model"),
        }
