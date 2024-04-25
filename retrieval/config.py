from __future__ import annotations

from typing import Callable

from data_utils import Movie
from retrieval.indexing_pipeline_utils import get_synopsys_txt_2
from retrieval.retrieval_pipeline_utils import clean_query_txt


class RetrievalExpsConfig:
    """
    Class to keep track of all the parameters used in the embeddings experiments.
    Any attribute created in this class will be logged to mlflow.

    Nota: cuando definimos atributos de tipo Callable, debemos usar `staticmethod` para que la función pueda ser llamada
    s
    """

    # SOME MODELS Y TRY 
    # models - "sentence-transformers/multi-qa-distilbert-cos-v1"
    # models - "sentence-transformers/all-MiniLM-L6-v2"
    # models - "sentence-transformers/all-mpnet-base-v2"
    # models - "mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es"
    # models - "hiiamsid/sentence_similarity_spanish_es"
    # models - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # models - "sentence-transformers/somosnlp-hackathon-2022/paraphrase-spanish-distilroberta"
    # models - "jinaai/jina-embeddings-v2-base-es"
    
    def __init__(self):

        # Función a emplear para generar el texto a indexar con embeddings; Debe tomar como input un objeto `Movie` y devolver un string
        self._text_to_embed_fn: Callable = get_synopsys_txt_2

        # Parámetros para la generación de embeddings

        self.model_name: str = "hiiamsid/sentence_similarity_spanish_es"
        self.normalize_embeddings: bool = False  # Normalizar los embeddings a longitud 1 antes de indexarlos

        self._query_prepro_fn: Callable = clean_query_txt

    ## NO MODIFICAR A PARTIR DE AQUÍ ##

    def text_to_embed_fn(self, movie: Movie) -> str:
        return self._text_to_embed_fn(movie)

    def query_prepro_fn(self, query: dict) -> str:
        return self._query_prepro_fn(query)

    @property
    def index_config_unique_id(self) -> str:
        return f"{self.model_name}_{self._text_to_embed_fn.__name__}_{self.normalize_embeddings}"

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
        }
