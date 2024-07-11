from __future__ import annotations

from typing import Callable

from data_utils import Movie
from retrieval.indexing_pipeline_utils import get_synopsys_txt, get_synopsys_txt_v2, get_synopsys_txt_v3, get_synopsys_txt_v4, get_synopsys_txt_v5
from retrieval.retrieval_pipeline_utils import clean_query_txt, clean_query_txt_v2, clean_query_txt_v3

class RetrievalExpsConfig:
    """
    Class to keep track of all the parameters used in the embeddings experiments.
    Any attribute created in this class will be logged to mlflow.

    Nota: cuando definimos atributos de tipo Callable, debemos usar `staticmethod` para que la función pueda ser llamada
    s
    """

    def __init__(self):

        # Función a emplear para generar el texto a indexar con embeddings; Debe tomar como input un objeto `Movie` y devolver un string
        self._text_to_embed_fn: Callable = get_synopsys_txt_v4

        # Parámetros para la generación de embeddings

        # self.model_name: str = "all-MiniLM-L6-v2"
        # self.model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
        # self.model_name: str = "intfloat/multilingual-e5-small"
        # self.model_name: str = "distiluse-base-multilingual-cased-v2"
        # self.model_name: str = "LaBSE"
        # self.model_name: str = "intfloat/multilingual-e5-base"
        # self.model_name: str = "paraphrase-multilingual-mpnet-base-v2"
        self.model_name: str = "hiiamsid/sentence_similarity_spanish_es"
        self.normalize_embeddings: bool = False  # Normalizar los embeddings a longitud 1 antes de indexarlos

        self._query_prepro_fn: Callable = clean_query_txt_v3

    ## NO MODIFICAR A PARTIR DE AQUÍ ##

    def text_to_embed_fn(self, movie: Movie) -> str:
        return self._text_to_embed_fn(movie)

    def query_prepro_fn(self, query: dict) -> str:
        return self._query_prepro_fn(query)

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
        }
