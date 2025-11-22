import os
from dotenv import load_dotenv
from vectorize_book import vectorize_book_and_store_to_db, vectorize_chapters

load_dotenv()

CLASS_SUBJECT_NAME = os.getenv('CLASS_SUBJECT_NAME')


vectorize_book_and_store_to_db(
    # CLASS_SUBJECT_NAME, "class_12_biology_vector_db"
    # CLASS_SUBJECT_NAME, "class_12_chemistry_vector_db"
    CLASS_SUBJECT_NAME, "class_12_physics_vector_db"
)
vectorize_chapters(CLASS_SUBJECT_NAME)