"""transformer-internals — encoder vs decoder, one layer at a time.

Run:  .venv/bin/python app.py        -> http://127.0.0.1:8310
"""
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import internals

log = logging.getLogger("transformer-internals")
app = FastAPI(title="transformer-internals")
STATIC = Path(__file__).parent / "static"


class TextBody(BaseModel):
    model: str
    text: str


class LayerBody(TextBody):
    layer: int


class ProjBody(LayerBody):
    token: int


class RowBody(LayerBody):
    head: int
    query: int


class TaskBody(TextBody):
    temperature: float = 1.0
    top_k: int = 50
    n_new: int = 8


def _run(fn):
    try:
        return fn()
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:                                   # noqa: BLE001
        log.exception("payload failed")
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def api_meta():
    return internals.meta_payload()


@app.post("/api/tokenize")
def api_tokenize(b: TextBody):
    return _run(lambda: internals.tokenize_payload(b.model, b.text))


@app.post("/api/forward")
def api_forward(b: TextBody):
    return _run(lambda: internals.forward_payload(b.model, b.text))


@app.post("/api/projections")
def api_projections(b: ProjBody):
    return _run(lambda: internals.projections_payload(
        b.model, b.text, b.layer, b.token))


@app.post("/api/row")
def api_row(b: RowBody):
    return _run(lambda: internals.row_payload(
        b.model, b.text, b.layer, b.head, b.query))


@app.post("/api/assembly")
def api_assembly(b: LayerBody):
    return _run(lambda: internals.assembly_payload(b.model, b.text, b.layer))


@app.post("/api/task")
def api_task(b: TaskBody):
    return _run(lambda: internals.task_payload(
        b.model, b.text, b.temperature, b.top_k, b.n_new))


app.mount("/", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s — %(message)s")
    log.info("ready — UI at http://127.0.0.1:8310/")
    uvicorn.run(app, host="127.0.0.1", port=8310)
