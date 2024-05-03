import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    # Connect to PostgreSQL
    connection = psycopg2.connect(
        host=os.environ['POSTGRES_HOST'],
        database=os.environ['POSTGRES_DATABASE'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD']
    )
    return connection
