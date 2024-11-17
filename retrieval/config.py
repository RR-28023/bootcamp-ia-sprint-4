from __future__ import annotations

from typing import Callable

from data_utils import Movie
#añadir todas las funciones creadas
from retrieval.indexing_pipeline_utils import get_synopsys_txt, getMovieData
from retrieval.retrieval_pipeline_utils import clean_query_txt, mod_query


class RetrievalExpsConfig:
    """
    Class to keep track of all the parameters used in the embeddings experiments.
    Any attribute created in this class will be logged to mlflow.

    Nota: cuando definimos atributos de tipo Callable, debemos usar `staticmethod` para que la función pueda ser llamada
    s
    """

    def __init__(self):

        # Función a emplear para generar el texto a indexar con embeddings; Debe tomar como input un objeto `Movie` y devolver un string
        self._text_to_embed_fn: Callable = getMovieData

        self.normalize_embeddings: bool = False  # Normalizar los embeddings a longitud 1 antes de indexarlos

        self._query_prepro_fn: Callable = clean_query_txt

        # Parámetros para la generación de embeddings

        # Definición del modelo
        # modelo empleado, se puede usar cualquiera de hugging face
        self.model_name: str = "mrm8488/multilingual-e5-large-ft-sts-spanish-matryoshka-768-64-5e"
        #self.model_name: str = "Shaharyar6/finetuned_sentence_similarity_spanish" 

        #self.model_name: str = "all-MiniLM-L6-v2"  
        #self.model_name: str = "all-mpnet-base-v2" 
        #self.model_name: str = "all-distilroberta-v1"  
        #self.model_name: str = "paraphrase-multilingual-mpnet-base-v2"  
        #self.model_name: str = "multi-qa-mpnet-base-dot-v1"  
        #self.model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  
        #self.model_name: str = "hiiamsid/sentence_similarity_spanish_es"  
        #self.model_name: str = "somosnlp-hackathon-2022/paraphrase-spanish-distilroberta"
        #self.model_name: str = "prudant/lsg_4096_sentence_similarity_spanish"
      

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
