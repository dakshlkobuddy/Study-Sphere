import os


def run_startup_checks(unified_db_path):
    checks = []
    groq_present = bool(os.getenv("GROQ_API_KEY", "").strip())
    checks.append(
        {
            "name": "GROQ_API_KEY",
            "ok": groq_present,
            "message": "Configured" if groq_present else "Missing GROQ_API_KEY",
        }
    )

    db_exists = os.path.isdir(unified_db_path)
    checks.append(
        {
            "name": "UNIFIED_VECTOR_DB",
            "ok": db_exists,
            "message": "Found" if db_exists else f"Missing path: {unified_db_path}",
        }
    )
    return checks


def all_checks_ok(checks):
    return all(check["ok"] for check in checks)

