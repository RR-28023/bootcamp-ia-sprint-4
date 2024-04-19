from __future__ import annotations
from typing import Callable
from data_utils import Movie
from retrieval.indexing_pipeline_utils import get_synopsys_txt
from retrieval.retrieval_pipeline_utils import clean_query_txt
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import DistilBertTokenizer, DistilBertModel
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import ElectraModel, ElectraTokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLMRobertaTokenizer
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from sentence_transformers import SentenceTransformer

class RetrievalExpsConfig:
    # Otros métodos de la clase aquí...

    @staticmethod
    def generate_embeddings(texts):
        # Definir el modelo y el tokenizador base
        base_model_name = "xlm-roberta-base"  # Cambia al modelo base si es diferente
        base_tokenizer = XLMRobertaTokenizer.from_pretrained(base_model_name)
        base_model = X

import torch
import random


class RetrievalExpsConfig:
    """
    Class to keep track of all the parameters used in the embeddings experiments.
    Any attribute created in this class will be logged to mlflow.

    Nota: cuando definimos atributos de tipo Callable, debemos usar `staticmethod` para que la función pueda ser llamada
    s
    """

    """ def __init__(self):

        # Función a emplear para generar el texto a indexar con embeddings; Debe tomar como input un objeto `Movie` y devolver un string
        self._text_to_embed_fn: Callable = get_synopsys_txt

        # Parámetros para la generación de embeddings

        self.model_name: str = "all-MiniLM-L6-v2"
        self.normalize_embeddings: bool = False  # Normalizar los embeddings a longitud 1 antes de indexarlos

        self._query_prepro_fn: Callable = clean_query_txt"""
   
    """ #EXPERIMENTO 1
    def __init__(self):

        # Función para generar el texto a indexar con embeddings
        self._text_to_embed_fn: Callable = get_synopsys_txt

        # Parámetros para la generación de embeddings
        self.model_name: str = "paraphrase-multilingual-mpnet-base-v2"
        self.normalize_embeddings: bool = False

        # Función para procesar la query de entrada
        self._query_prepro_fn: Callable = clean_query_txt"""
    
    """"
    EXPERIMENTO 2
    
    def __init__(self):

        # Función para generar el texto a indexar con embeddings
        self._text_to_embed_fn: Callable = get_synopsys_txt

        # Parámetros para la generación de embeddings
        self.model_name: str = "xlm-r-bert-base-nli-stsb-mean-tokens"  # Cambiamos el modelo
        self.normalize_embeddings: bool = True  # Normalizar los embeddings a longitud 1 antes de indexarlos

        # Función para procesar la query de entrada
        self._query_prepro_fn: Callable = clean_query_txt """
    
    
    
    """def __init__(self):
        # Función para generar el texto a indexar con embeddings
        self._text_to_embed_fn: Callable = get_synopsys_txt

        # Parámetros para la generación de embeddings
        self.model_name: str = "xlm-roberta-base"
        self.normalize_embeddings: bool = True

        # Función para procesar la query de entrada
        self._query_prepro_fn: Callable = clean_query_txt"""
    

    """def __init__(self):
        # Función para generar el texto a indexar con embeddings
        self._text_to_embed_fn: Callable = get_synopsys_txt

        # Parámetros para la generación de embeddings
        self.model_name: str =  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.normalize_embeddings: bool = True

        # Función para procesar la query de entrada
        self._query_prepro_fn: Callable = clean_query_txt"""
    

    """def __init__(self):
        # Función para generar el texto a indexar con embeddings
        self._text_to_embed_fn: Callable = get_synopsys_txt

        # Parámetros para la generación de embeddings
        self.model_name: str = "google/electra-small-discriminator"
        model = ElectraModel.from_pretrained(self.model_name)
        self.normalize_embeddings: bool = True

        # Función para procesar la query de entrada
        self._query_prepro_fn: Callable = clean_query_txt"""
    

    """def __init__(self):
        # Función para generar el texto a indexar con embeddings
        self._text_to_embed_fn: Callable = get_synopsys_txt

        # Parámetros para la generación de embeddings
        self.model_name: str = "bert-base-uncased"  
        self.normalize_embeddings: bool = True  # Normalizamos los embeddings

        # Función para procesar la query de entrada
        self._query_prepro_fn: Callable = clean_query_txt

        # Hiperparámetros para el ajuste fino del modelo
        self.learning_rate: float = 2e-5
        self.batch_size: int = 32
        self.num_epochs: int = 3

    @staticmethod
    def generate_embeddings(texts):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

        # Tokenizar los textos
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Obtener los embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Tomar la representación del token CLS para cada texto
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        return embeddings



        
    @staticmethod
    def generate_embeddings(texts):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
        tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")

        # Tokenizar los textos
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Obtener los embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Tomar la representación del token CLS para cada texto
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        return embeddings"""
    
 

    """def __init__(self):
        # Función para generar el texto a indexar con embeddings
        self._text_to_embed_fn = get_synopsys_txt

        # Parámetros para la generación de embeddings
        self.model_name = "google/electra-small-generator"  # Utilizamos el modelo ELECTRA base
        self.normalize_embeddings = True  # Normalizamos los embeddings

        # Función para procesar la query de entrada
        self._query_prepro_fn = clean_query_txt

        # Hiperparámetros para el ajuste fino del modelo
        self.learning_rate = 1e-5  # Reducimos el learning rate
        self.batch_size = 64  # Aumentamos el batch size
        self.num_epochs = 30  # Aumentamos el número de épocas

        # Establecemos una semilla aleatoria para reproducibilidad
        random.seed(42)"""
    


    """@staticmethod
    def generate_embeddings(texts):
        tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-generator")
        model = ElectraModel.from_pretrained("google/electra-small-generator")

        # Tokenizar los textos
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Obtener los embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Tomar la representación del token CLS para cada texto
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        return embeddings"""
    

    """def __init__(self):
        # Función para generar el texto a indexar con embeddings
        self._text_to_embed_fn = get_synopsys_txt

        # Parámetros para la generación de embeddings
        self.model_name = "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking"  
        self.normalize_embeddings = True  # Normalizamos los embeddings

        # Función para procesar la query de entrada
        self._query_prepro_fn = clean_query_txt

        # Hiperparámetros para el ajuste fino del modelo
        self.learning_rate = 1e-5  # Reducimos el learning rate
        self.batch_size = 13  # Reducimos el tamaño del lote
        self.num_epochs = 20  # Aumentamos el número de épocas

        # Establecemos una semilla aleatoria para reproducibilidad
        random.seed(42)

    @staticmethod
    def generate_embeddings(texts):
        tokenizer = XLMRobertaModel.from_pretrained("sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking")
        model = XLMRobertaTokenizer.from_pretrained("sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking")

        # Tokenizar los textos
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Obtener los embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Tomar la representación del token CLS para cada texto
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        return embeddings"""
    

    """def __init__(self):
        # Función para generar el texto a indexar con embeddings
        self._text_to_embed_fn = get_synopsys_txt

        # Parámetros para la generación de embeddings
        self.model_name = "igorsterner/xlmr-multilingual-sentence-segmentation"  
        self.normalize_embeddings = True  # Normalizamos los embeddings

        # Función para procesar la query de entrada
        self._query_prepro_fn = clean_query_txt

        # Hiperparámetros para el ajuste fino del modelo
        self.learning_rate = 3e-5  # Aumentamos ligeramente el learning rate
        self.batch_size = 32  # Aumentamos el tamaño del lote
        self.num_epochs = 50  # Aumentamos el número de épocas

        # Establecemos una semilla aleatoria para reproducibilidad
        random.seed(42)

    @staticmethod
    def generate_embeddings(texts):
        tokenizer = XLMRobertaModel.from_pretrained("igorsterner/xlmr-multilingual-sentence-segmentation")
        model = XLMRobertaTokenizer.from_pretrained("igorsterner/xlmr-multilingual-sentence-segmentation")

        # Tokenizar los textos
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Obtener los embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Tomar la representación del token CLS para cada texto
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        return embeddings"""
    

    def __init__(self, initialize_model_weights=True):
        # Función para generar el texto a indexar con embeddings
        self._text_to_embed_fn = get_synopsys_txt

        # Parámetros para la generación de embeddings
        self.model_name = "embaas/sentence-transformers-multilingual-e5-base"  
        self.normalize_embeddings = True  # Normalizamos los embeddings

        # Función para procesar la query de entrada
        self._query_prepro_fn = clean_query_txt

        # Hiperparámetros para el ajuste fino del modelo
        self.learning_rate = 3e-5  # Aumentamos ligeramente el learning rate
        self.batch_size = 32  # Aumentamos el tamaño del lote
        self.num_epochs = 50  # Aumentamos el número de épocas

        # Establecemos una semilla aleatoria para reproducibilidad
        random.seed(42)

        # Parámetro para la inicialización de pesos
        self.initialize_model_weights = initialize_model_weights


    @staticmethod
    def generate_embeddings(texts, initialize_model_weights=True):
        tokenizer = XLMRobertaModel.from_pretrained("embaas/sentence-transformers-multilingual-e5-base")
        
        if initialize_model_weights:
            model = XLMRobertaModel.from_pretrained("embaas/sentence-transformers-multilingual-e5-base")
        else:
            model = XLMRobertaModel.from_pretrained("embaas/sentence-transformers-multilingual-e5-base", 
                                                     init_weights=False)

        # Tokenizar los textos
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Obtener los embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Tomar la representación del token CLS para cada texto
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        return embeddings
    


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

