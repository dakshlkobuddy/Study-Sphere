import os
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "vector_db" / "class_12_unified_vector_db"
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if not groq_key:
        print("healthcheck: missing GROQ_API_KEY")
        return 1
    if not db_path.exists():
        print(f"healthcheck: missing vector db at {db_path}")
        return 1
    print("healthcheck: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

