from __future__ import annotations

from typing import Callable

from data_utils import Movie
from retrieval.indexing_pipeline_utils import get_synopsys_txt
from retrieval.retrieval_pipeline_utils import clean_query_txt
from retrieval.indexing_pipeline_utils import combined_text
from retrieval.retrieval_pipeline_utils import advanced_clean_query_txt
from retrieval.indexing_pipeline_utils import get_detailed_txt
from retrieval.retrieval_pipeline_utils import advanced_clean_query_txt2

from retrieval.indexing_pipeline_utils import (
    get_synopsys_txt,
    get_title_and_genre_txt,
    get_director_and_cast_txt
)
from retrieval.retrieval_pipeline_utils import (
    clean_query_txt,
    preprocess_query_no_el_usuario,
    preprocess_query_lowercase
)

class RetrievalExpsConfig:
    """
    Class to keep track of all the parameters used in the embeddings experiments.
    Any attribute created in this class will be logged to mlflow.

    Nota: cuando definimos atributos de tipo Callable, debemos usar `staticmethod` para que la función pueda ser llamada
    s
    """

    def __init__(self):

        self._text_to_embed_fn: Callable = get_detailed_txt 
        self.model_name: str = "sentence-transformers/all-mpnet-base-v2"  
        self.normalize_embeddings: bool = True  
        self._query_prepro_fn: Callable = advanced_clean_query_txt2  

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
