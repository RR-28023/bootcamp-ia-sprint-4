import json
import os

import psycopg
from dotenv import load_dotenv

from .schemas import Movie


class FilmRecDbConn:
    """
    Connection to the film_rec database, currently in a local postgres instance.
    """

    def __init__(self):

        load_dotenv()
        dbuser, dbname = os.getenv("RDS_USER"), os.getenv("RDS_DB")
        dbhost, dbpwd = os.getenv("RDS_HOST"), os.getenv("RDS_PW")
        self.ps_conn = psycopg.connect(user=dbuser, dbname=dbname, host=dbhost, password=dbpwd)

    def begin_transaction(self):
        self.ps_conn.execute("BEGIN")

    def commit_transaction(self):
        self.ps_conn.commit()

    def execute_wo_commit(self, query: str):
        self.ps_conn.execute(query)

    def rollback_transaction(self):
        self.ps_conn.rollback()

    def run_insert_query(self, query: str):
        self.ps_conn.execute(query)
        self.ps_conn.commit()

    def run_read_query(self, query: str) -> tuple:
        return self.ps_conn.execute(query).fetchall()

    def close(self):
        self.ps_conn.close()


def get_movies_data() -> list[Movie]:
    """
    Returns all movies data from the movie_attributes table.
    """
    json_path = "retrieval/data/movies_data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        movies_data = json.load(f)
    
    # Depuración: Imprimir los datos cargados
    for movie in movies_data[:5]:  # Muestra los primeros 5 para no saturar la salida
        print(f"Cargado: {movie}")
    
    movies = [
        Movie(
            movie_id=movie["movie_id"],
            title_es=movie["title_es"],
            title_original=movie["title_original"],
            duration_mins=movie["duration_mins"],
            year=movie["year"],
            country=movie["country"],
            genre_tags=movie["genre_tags"],
            tv_show_flag=movie["tv_show_flag"],
            director_top_5=movie["director_top_5"],
            script_top_5=movie["script_top_5"],
            cast_top_5=movie["cast_top_5"],
            photography_top_5=movie["photography_top_5"],
            synopsis=movie["synopsis"],
        )
        for movie in movies_data
    ]
    return movies



def get_movies_data_from_db(conn: FilmRecDbConn = None) -> list[Movie]:
    """
    Returns all movies data from the movie_attributes table.
    Aplicamos unos filtros para reducir el número de películas y que sea más manejable para el ejercicio.
    """
    conn = conn or FilmRecDbConn()
    query = f"""
        SELECT 
            movie_id, 
            title_es,
            title_original,
            duration_mins,
            year,
            country,
            genre_tags,
            tv_show_flag,
            director_top_5,
            script_top_5,
            cast_top_5,
            photography_top_5,
            synopsis
        FROM movie_attributes  
        WHERE synopsis != '' -- NO CAMBIAR ESTOS FILTROS!
            AND "year" > 2010 
            AND tv_show_flag IS FALSE 
            AND duration_mins > 60
    """
    movie_data = conn.run_read_query(query)
    movies = [
        Movie(
            movie_id=movie[0],
            title_es=movie[1],
            title_original=movie[2],
            duration_mins=movie[3],
            year=movie[4],
            country=movie[5],
            genre_tags=movie[6],
            tv_show_flag=movie[7],
            director_top_5=movie[8],
            script_top_5=movie[9],
            cast_top_5=movie[10],
            photography_top_5=movie[11],
            synopsis=movie[12],
        )
        for movie in movie_data
    ]
    return movies
