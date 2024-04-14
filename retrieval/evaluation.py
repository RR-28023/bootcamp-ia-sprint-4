import json
import random
import matplotlib.pyplot as plt
import seaborn as sns


from langchain_core.documents import Document

from data_utils import Movie
from retrieval.config import RetrievalExpsConfig


def generar_prompts_evaluation_set(movies: list[Movie]):
    """
    Prompts que le pasamos a GPT-4 para que genere una lista de 100 películas a evaluar.
    """
    # Sort by movie_id
    movies = sorted(movies, key=lambda x: x.movie_id)

    # Sample 100 movies randomly without replacement
    random.seed(42)
    sample = random.sample(movies, 350)  # 300 para evaluación, 50 para test
    prompt = """ 
    Eres un científico de datos y te han encargado evaluar un modelo de recomendación de películas, en concreto estamos
    evaluando la calidad del módulo de retrieval usando retrieval Para ello, para cada película de la lista necesitamos
    generar ejemplos de queries que deberían retornar esa película.
    Las queries deben ser: 
    - Frases o párrafos de entre 10 y 50 palabras
    - Deben asociarse a la película, pero pueden ser suficientemente genéricas para que no SÓLO incluya a esa película
    - No deben parafrasear directamente la sinopsis, tiene que ser un poco dificil para el modelo!!
    - Todas las queries deben empezar por: "El usuario busca" 
    \n
    Por favor, genera las queries como un JSON donde se incluya el movie_id.
    \n
    Por ejemplo:
    # Película
    {'movie_id': 100119, 'title_es': 'Perdona que te moleste', 'duration_mins': 110, 'country': 'Estados Unidos', 'genre_tags': 'Comedia;Fantástico;Trabajo/empleo', 'synopsis': 'Un vendedor telefónico con problemas de autoestima descubre la clave para el éxito en el negocio. Pero cuando empieza a subir escalones en la compañía, sus amigos activistas denuncian prácticas laborales injustas. (FILMAFFINITY)'}
    # Query
    {
        "movie_id": 100119,
        "query": "El usuario busca una comedia en un entorno laboral, que trate temas como el ascenso profesional y la ética profesional."
    }
    \n
    # Película
    {'movie_id': 100213, 'title_es': 'El Príncipe', 'duration_mins': 92, 'country': 'Chile', 'genre_tags': 'Drama;Años 70;Homosexualidad;Drama carcelario;Crimen', 'synopsis': 'San Bernardo, Chile, justo antes que Allende asuma la presidencia, en una noche de borrachera, Jaime, un joven de 20 años solitario y narcisista acuchilla a su mejor amigo, El Gitano, en un aparente arrebato pasional. En la cárcel conoce a El Potro, un hombre mayor y respetado a quien se acerca necesitado de protección, ternura y reconocimiento. Jaime se convierte en El Príncipe y descubre el amor y la lealtad mientras asiste a la violenta lucha de poder en la prisión. (FILMAFFINITY)'}
    # Query
    {
        "movie_id": 100213,
        "query": "El usuario busca una película de habla hispana, idealmente un drama carcelario que trate temas como la amistad".
    }
    \n
    # Película
    {'movie_id': 100213, 'title_es': 'El Príncipe', 'duration_mins': 92, 'country': 'Chile', 'genre_tags': 'Drama;Años 70;Homosexualidad;Drama carcelario;Crimen', 'synopsis': 'San Bernardo, Chile, justo antes que Allende asuma la presidencia, en una noche de borrachera, Jaime, un joven de 20 años solitario y narcisista acuchilla a su mejor amigo, El Gitano, en un aparente arrebato pasional. En la cárcel conoce a El Potro, un hombre mayor y respetado a quien se acerca necesitado de protección, ternura y reconocimiento. Jaime se convierte en El Príncipe y descubre el amor y la lealtad mientras asiste a la violenta lucha de poder en la prisión. (FILMAFFINITY)'}
    # Query
    {
        "movie_id": 100213,
        "query": "El usuario busca una película no muy larga (menos de 120 mins) que mezcle cárceles, violencia y pasión".
    }
    \n
    # Película
    {'movie_id': 100469, 'title_es': 'El color púrpura', 'duration_mins': 141, 'country': 'Estados Unidos', 'genre_tags': 'Musical;Drama;Feminismo;Racismo;Años 1900 (circa);Años 1910-1919;Años 20', 'synopsis': 'En 1909, Celie, una chica negra norteamericana, es entregada en matrimonio por su maltratador padre a un granjero local, Albert, que la trata con crueldad. Celie es temerosa de Dios y su liberación llega en forma de una cantante de jazz. Adaptación de la novela de Alice Walker sobre las luchas de toda la vida de una mujer afroamericana a principios del siglo XX.'}
    # Query
    {
        "movie_id": 100469,
        "query": "El usuario busca un musical que trate temas como el racismo, la violencia familiar y la fe. Idealmente basada en una novela."
    }
    \n
    # Película
    {'movie_id': 100554, 'title_es': 'La corresponsal', 'duration_mins': 105, 'country': 'Estados Unidos', 'genre_tags': 'Bélico;Acción;Drama;Biográfico;Periodismo;Guerra de Siria;Basado en hechos reales', 'synopsis': 'Marie Colvin (Rosamund Pike) era una periodista reconocida mundialmente por su trabajo en distintos conflictos bélicos. Testigo de algunas cruentas batallas recientes, especialmente de lo sucedido en Oriente Medio, contaba con el respeto tanto de los lectores como de sus compañeros de profesión por su enorme valentía y humildad. Sin embargo, su personalidad era caótica y autodestructiva. Tras recibir el impacto de una granada en Sri Lanka, comienza a llevar un distintivo parche en el ojo mientras se sienta a beber rodeada de la alta sociedad londinense a la que aborrece, hasta que un día recibe una misión extremadamente peligrosa, que acepta junto a su prestigioso fotógrafo de guerra, Paul Conroy (Jamie Dornan). Juntos viajan a Siria para cubrir lo que sucede en la ciudad de Homs, donde aprenderá el verdadero coste de la guerra, tanto física como psicológicamente. (FILMAFFINITY)'}
    # Query
    {
        "movie_id": 100554,
        "query": "El usuario busca una película sobre la guerra de Siria, basada en hechos reales."
    }
    \n
    # Película
    {'movie_id': 828416, 'title_es': 'El lobo de Wall Street', 'duration_mins': 179, 'country': 'Estados Unidos', 'genre_tags': 'Comedia;Drama;Comedia negra;Biográfico;Bolsa & Negocios;Años 80;Años 90;Drogas', 'synopsis': 'Película basada en hechos reales del corredor de bolsa neoyorquino Jordan Belfort (Leonardo DiCaprio). A mediados de los años 80, Belfort era un joven honrado que perseguía el sueño americano, pero pronto en la agencia de valores aprendió que lo más importante no era hacer ganar a sus clientes, sino ser ambicioso y ganar una buena comisión. Su enorme éxito y fortuna le valió el mote de “El lobo de Wall Street”. Dinero. Poder. Mujeres. Drogas. Las tentaciones abundaban y el temor a la ley era irrelevante. Jordan y su manada de lobos consideraban que la discreción era una cualidad anticuada; nunca se conformaban con lo que tenían. (FILMAFFINITY)'}
    # Query
    {
        "movie_id": 828416,
        "query": "El usuario busca una película basada en hechos reales sobre dinero, drogas y finanzas."
    }
    \n
    Ahora por favor, genera queries para las siguientes películas:
    \n
    """
    movie_as_txt = lambda m: str(
        m.model_dump(
            exclude=[
                "title_original",
                "year",
                "director_top_5",
                "script_top_5",
                "cast_top_5",
                "photography_top_5",
                "tv_show_flag",
            ]
        )
    )

    for i in range(6):
        # Generamos 6  + 1 (de test) prompts diferentes para obtener mejores resultados con GPT-4 (en vez de un prompt muy largo)
        prompt_to_save = prompt + "\n".join([movie_as_txt(m) for m in sample[i * 50 : (i + 1) * 50]])
        with open(f"evaluation_set_prompt_{i+1}.txt", "w") as f:
            f.write(prompt_to_save)

    return


def load_eval_queries():
    with open("retrieval/evaluation/data/eval_queries.json", "r") as f:
        queries = json.load(f)
    return queries


def load_test_queries():
    with open("retrieval/evaluation/data/test_queries.json", "r") as f:
        queries = json.load(f)
    return queries


def calc_mrr(expected_movie_id: int, retrieved_movies_ids=list[int]) -> tuple[float, int]:
    """
    Dada una query y una lista de películas devueltas usando nuestro algoritmo de retrieval, calculamos el MRR
    """
    mrr = 0 
    rank = -1
    if expected_movie_id in retrieved_movies_ids:
        rank = retrieved_movies_ids.index(expected_movie_id) + 1
        mrr += 1 / rank
    mrr = mrr / len(retrieved_movies_ids)
    return mrr, rank


def is_in_results(expected_movie_id: int, retrieved_movies_ids=list[int]):
    """
    Dada una query y una lista de películas devueltas usando nuestro algoritmo de retrieval, calculamos el MRR
    """
    return expected_movie_id in retrieved_movies_ids


def comparar_resultado_con_esperado(
    query: str, resultado: Document, esperado: Document, exp_config: RetrievalExpsConfig
) -> str:
    txt_to_embed_fn = exp_config.text_to_embed_fn
    # Convertir el resultado a objetos movie
    resultado = Movie(**resultado.metadata)
    esperado = Movie(**esperado.metadata)

    comp = f""" 
-----------------------------
## Query:
{query}
## Doc resultado obtenido:
{txt_to_embed_fn(resultado)}
## Doc resultado esperado:
{txt_to_embed_fn(esperado)}
-----------------------------
    """
    return comp

def plot_rank_distribution(ranks = list[int]):
    """ 
    Draw a chart showing the histogram of ranks for the correct movie in the retrieval results.
    The chart will have 11 bars, 10 bars for the positions 1-10 and 1 bar for when the movie was not found in the top 10.
    Returns the figure object.
    """
    x_lables = [str(i) for i in range(1, 11)] + ["No encontrada"]
    y_values = [ranks.count(i) for i in range(1, 11)]
    y_values.append(len(ranks) - sum(y_values))
    plt.figure(figsize=(12, 6))
    sns.barplot(x=x_lables, y=y_values, color="skyblue")
    plt.title("Posiciones de la película buscada en los resultados de retrieval")
    for i, v in enumerate(y_values):
        plt.text(i, v, str(v), ha='center', va='bottom')
    return plt.gcf()