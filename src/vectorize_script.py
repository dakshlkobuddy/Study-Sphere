import argparse

from vectorize_book import (
    vectorize_book_and_store_to_db,
    vectorize_chapters,
    vectorize_unified_db,
)


subjects = ["physics", "chemistry", "biology"]
vector_db_names = {
    "physics": "class_12_physics_vector_db",
    "chemistry": "class_12_chemistry_vector_db",
    "biology": "class_12_biology_vector_db",
}


def run_legacy_vectorization():
    print("Starting legacy vectorization for all subjects")
    for subject in subjects:
        print("==============================")
        print(f"SUBJECT: {subject.upper()}")
        print("==============================")
        vectorize_book_and_store_to_db(subject, vector_db_names[subject])
        vectorize_chapters(subject)
    print("Legacy vectorization completed")


def main():
    parser = argparse.ArgumentParser(description="Vectorization utility for Study Sphere")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Build legacy subject/chapter vector DB structure",
    )
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Build unified vector DB with subject/chapter/page metadata",
    )
    parser.add_argument(
        "--recreate-unified",
        action="store_true",
        help="Delete and recreate unified vector DB",
    )
    args = parser.parse_args()

    run_unified = args.unified or not args.legacy

    if args.legacy:
        run_legacy_vectorization()

    if run_unified:
        print("Starting unified vectorization")
        vectorize_unified_db(subjects=subjects, recreate=args.recreate_unified)
        print("Unified vectorization completed")


if __name__ == "__main__":
    main()
