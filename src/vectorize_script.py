from vectorize_book import vectorize_book_and_store_to_db, vectorize_chapters

subjects = ["physics", "chemistry", "biology"]

vector_db_names = {
    "physics": "class_12_physics_vector_db",
    "chemistry": "class_12_chemistry_vector_db",
    "biology": "class_12_biology_vector_db"
}

print("\n🔄 Starting vectorization for ALL subjects...\n")

for subject in subjects:
    print(f"==============================")
    print(f"📘 SUBJECT: {subject.upper()}")
    print(f"==============================")

    # 1. Vectorize complete book
    vectorize_book_and_store_to_db(subject, vector_db_names[subject])

    # 2. Vectorize each chapter
    vectorize_chapters(subject)

print("\n✅ All subjects have been vectorized successfully!\n")