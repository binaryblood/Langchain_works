import os
import csv

from openai import OpenAI
from neo4j import GraphDatabase, basic_auth

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_movie_tags(limit=None):
    driver = GraphDatabase.driver(os.getenv("NEO4J_SANDBOX_URI"),auth=basic_auth(os.getenv("NEO4J_SANDBOX_USERNAME"), os.getenv("NEO4J_SANDBOX_PASSWORD")))

    driver.verify_connectivity()
    print("Connection verified!")
    query = """MATCH (m:Movie) WHERE m.tagline IS NOT NULL
    RETURN m.movieId AS movieId, m.title AS title, m.tagline AS tagline"""

    if limit is not None:
        query += f' LIMIT {limit}'
    print("Executing: "+str(query))
    movies, summary, keys = driver.execute_query(
        query
    )
    print("results: "+str(movies))
    driver.close()

    return movies

def generate_embeddings(file_name, limit=None):

    csvfile_out = open(file_name, 'w', encoding='utf8', newline='')
    fieldnames = ['movieId','embedding']
    output_plot = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
    output_plot.writeheader()

    movies = get_movie_tags(limit=limit)

    print(len(movies))
    
    llm = OpenAI()

    for movie in movies:
        print(movie['title'])

        tagline = f"{movie['title']}: {movie['tagline']}"
        response = llm.embeddings.create(
            input=tagline,
            model='text-embedding-ada-002'
        )

        output_plot.writerow({
            'movieId': movie['movieId'],
            'embedding': response.data[0].embedding
        })

    csvfile_out.close()

generate_embeddings('.\data\\movie-plot-embeddings.csv')
