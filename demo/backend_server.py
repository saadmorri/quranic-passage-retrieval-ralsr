from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import BenchmarkStore
from config import SYSTEMS, load_config
from data_loader import ResourceStore
from query_pipeline import QueryProcessor
from retrieval.bm25_retriever import BM25Retriever
from retrieval.crossencoder_reranker import CrossEncoderReranker
from retrieval.dense_e5_retriever import DenseE5Retriever
from retrieval.ralsr_retriever import RALSRRetriever


ZERO_CANDIDATE_MESSAGE = (
    "No RALSR candidate passage was found because no passage shared an accepted "
    "query root under the thesis candidate-generation rule."
)


class Services:
    def __init__(self):
        self.config = load_config()
        os.environ["CAMELTOOLS_DATA"] = str(self.config.camel_tools_data)
        self.resources = ResourceStore.load(self.config)
        self.benchmark = BenchmarkStore(self.resources)
        self.processor = QueryProcessor(self.resources.qac_lookup, self.resources.maqayis)
        self._bm25 = None
        self._dense = None
        self._ralsr = None
        self._ce = None
        self._lock = threading.RLock()

    @property
    def bm25(self):
        with self._lock:
            if self._bm25 is None:
                self._bm25 = BM25Retriever(self.resources.processed_qpc)
            return self._bm25

    @property
    def dense(self):
        with self._lock:
            if self._dense is None:
                self._dense = DenseE5Retriever(
                    self.resources.qpc,
                    self.config.e5_model_dir,
                    self.config.dense_embedding_cache,
                    self.config.dense_metadata_cache,
                )
            return self._dense

    @property
    def ralsr(self):
        with self._lock:
            if self._ralsr is None:
                self._ralsr = RALSRRetriever(
                    self.resources.passage_representation,
                    self.resources.normalization,
                )
            return self._ralsr

    @property
    def crossencoder(self):
        with self._lock:
            if self._ce is None:
                self._ce = CrossEncoderReranker(self.config.crossencoder_model_dir)
            return self._ce

    def free_query(self, question: str, system: str, top_n: int):
        if system not in SYSTEMS:
            raise ValueError(f"Unknown system: {system}")
        query = self.processor.process(question)
        candidate_count = None
        if system == "BM25":
            results = self.bm25.retrieve(query["selected_terms"], top_n)
        elif system == "Dense E5":
            results = self.dense.retrieve(question, top_n)
        elif system == "Fixed RALSR":
            results, candidate_count = self.ralsr.retrieve(query, top_n=top_n, candidate_depth=100)
        else:
            candidates, candidate_count = self.ralsr.candidates(query, depth=100)
            results = self.crossencoder.rerank(question, candidates, top_n=top_n) if candidates else []
        return {
            "mode": "live_free_query",
            "question": question,
            "system": system,
            "top_n": top_n,
            "query_processing": query,
            "candidate_count": candidate_count,
            "zero_candidate": system in {"Fixed RALSR", "RALSR + CrossEncoder"} and not results,
            "zero_candidate_message": ZERO_CANDIDATE_MESSAGE if system in {"Fixed RALSR", "RALSR + CrossEncoder"} and not results else None,
            "structural_output": "-1" if system in {"Fixed RALSR", "RALSR + CrossEncoder"} and not results else None,
            "results": results,
        }


SERVICES = None


def get_services():
    global SERVICES
    if SERVICES is None:
        SERVICES = Services()
    return SERVICES


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stdout.write("[backend] " + fmt % args + "\n")

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path == "/health":
                services = get_services()
                self._json(200, {"status": "ok", "questions": 251, "passages": 1266})
            elif self.path == "/questions":
                self._json(200, {"questions": get_services().benchmark.list_questions()})
            else:
                self._json(404, {"error": "Not found"})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/benchmark":
                result = get_services().benchmark.results(
                    int(payload["qid"]), str(payload["system"]), int(payload["top_n"])
                )
            elif self.path == "/free":
                result = get_services().free_query(
                    str(payload["question"]), str(payload["system"]), int(payload["top_n"])
                )
            else:
                self._json(404, {"error": "Not found"})
                return
            self._json(200, result)
        except Exception as exc:
            self._json(500, {"error": str(exc), "type": type(exc).__name__})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Step 21 backend listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
