from __future__ import annotations

from typing import Callable

from data_utils import Movie
from retrieval.indexing_pipeline_utils import get_synopsys_txt , get_sy_au_dir , get_sy_au_dir_year
from retrieval.retrieval_pipeline_utils import clean_query_txt_caracteres_especiales


class RetrievalExpsConfig:
    """
    Class to keep track of all the parameters used in the embeddings experiments.
    Any attribute created in this class will be logged to mlflow.

    Nota: cuando definimos atributos de tipo Callable, debemos usar `staticmethod` para que la función pueda ser llamada
    s
    """

    def __init__(self):

        # Función a emplear para generar el texto a indexar con embeddings; Debe tomar como input un objeto `Movie` y devolver un string

        #self._text_to_embed_fn: Callable = get_synopsys_txt
        self._text_to_embed_fn: Callable = get_sy_au_dir
        #self._text_to_embed_fn: Callable = get_sy_au_dir_year


        # Parámetros para la generación de embeddings
        
        #self.model_name: str = "all-MiniLM-L6-v2"
        #self.model_name: str = "bert-base-multilingual-cased"
        #self.model_name: str = "Qwen/CodeQwen1.5-7B-Chat"
        #self.model_name: str = "ecastera/eva-mistral-dolphin-7b-spanish"
        #self.model_name: str = "maidalun1020/bce-embedding-base_v1"
        #self.model_name: str = "Salesforce/SFR-Embedding-Mistral"        
        #self.model_name: str = "Snowflake/snowflake-arctic-embed-m"
        #self.model_name: str = "NeuralNovel/Gecko-7B-v0.1-DPO"
        #self.model_name: str = "acge_text_embedding"
        self.model_name: str = "intfloat/multilingual-e5-large"
    

        self.normalize_embeddings: bool = False  # Normalizar los embeddings a longitud 1 antes de indexarlos

        self._query_prepro_fn: Callable = clean_query_txt_caracteres_especiales

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
