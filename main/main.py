import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONPATH", str(ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("rfp_rag.api:app", host="0.0.0.0", port=8000, reload=True)
