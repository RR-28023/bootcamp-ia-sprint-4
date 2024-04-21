from pydantic import BaseModel

class Movie(BaseModel):
    movie_id: int
    title_es: str
    title_original: str
    duration_mins: int | None
    year: int
    country: str
    genre_tags: str
    tv_show_flag: bool
    director_top_5: str
    script_top_5: str
    cast_top_5: str
    photography_top_5: str
    synopsis: str

    def __repr__(self):
        return f"Movie(movie_id={self.movie_id}, title_es={self.title_es}, year={self.year})"

    def __str__(self):
        return self.__repr__()
    
    @property
    def url(self):
        return f"https://www.filmaffinity.com/es/film{self.movie_id}.html"