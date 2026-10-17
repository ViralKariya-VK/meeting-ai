import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="mai_tests_")
os.environ["STORAGE_DIR"] = _tmp
os.environ["EMBED_BACKEND"] = "hash"
os.environ["GROQ_API_KEY"] = ""
os.environ["APP_PASSWORD"] = "test-pass"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def clean_db():
    from services import config, db
    if config.DB_PATH.exists():
        for suffix in ("", "-wal", "-shm"):
            Path(str(config.DB_PATH) + suffix).unlink(missing_ok=True)
    db.init_db()
    yield
