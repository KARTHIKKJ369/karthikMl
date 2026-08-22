import json
import mimetypes
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, Optional

from mlx_lm import load as mlx_load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler as mlx_make_sampler

STATIC_DIR = Path(__file__).parent / "web"
ADAPTER_PATH = Path("checkpoints/mlx_karthik_adapters_1.5b")
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

_MODEL = None
_TOKENIZER = None


def get_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        print(f"Loading Karthik Jayan Portfolio SLM ({BASE_MODEL} + LoRA)...")
        _MODEL, _TOKENIZER = mlx_load(BASE_MODEL, adapter_path=str(ADAPTER_PATH))
    return _MODEL, _TOKENIZER


class PortfolioHTTPHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/models" or self.path == "/api/status":
            self._send_json({
                "status": "ready",
                "model": "Karthik Jayan Portfolio SLM (500M 4-bit)",
                "framework": "Apple MLX (Metal Performance Shaders)",
                "models": [
                    {"id": "karthik_portfolio", "name": "Karthik Jayan Portfolio SLM (500M)", "params": "500M (4-bit)", "context": 2048}
                ]
            })
            return

        req_path = self.path.split("?")[0].lstrip("/")
        if req_path == "" or req_path == "/":
            req_path = "index.html"

        file_path = STATIC_DIR / req_path
        if file_path.exists() and file_path.is_file():
            mime_type, _ = mimetypes.guess_type(file_path)
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, f"File {req_path} not found")

    def do_POST(self) -> None:
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)

            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception as e:
                self._send_json({"error": f"Invalid JSON payload: {str(e)}"}, status=400)
                return

            raw_prompt = payload.get("prompt", "Tell me about yourself.")
            temperature = float(payload.get("temperature", 0.2))
            top_p = float(payload.get("top_p", 0.9))
            top_k = int(payload.get("top_k", 40))
            max_new_tokens = int(payload.get("max_new_tokens", 250))

            try:
                model, tokenizer = get_model()

                # Clean prompt
                clean_question = raw_prompt.replace("### User:\n", "").replace("### Assistant:\n", "").replace("### System:\n", "").strip()
                if not clean_question:
                    clean_question = "Tell me about yourself."

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are Karthik Jayan, an AI Systems Engineer. Always answer in the first person as Karthik. "
                            "When asked about 'Ridge', 'Recall', 'CyberLabs', or any projects, refer to your software engineering systems (e.g., Ridge is your Self-Correcting RAG platform at ridge.karthikjayan.tech), never generic dictionary or geological definitions. "
                            "Stay strictly grounded in your verified experience, projects, and background from karthikjayan.dev."
                        )
                    },
                    {"role": "user", "content": clean_question},
                ]

                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                sampler = mlx_make_sampler(temp=max(temperature, 0.1), top_p=top_p, top_k=top_k)

                start_time = time.perf_counter()
                response_text = mlx_generate(
                    model,
                    tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_new_tokens,
                    sampler=sampler,
                )
                elapsed = time.perf_counter() - start_time

                tokens_generated = len(tokenizer.encode(response_text))
                tok_per_sec = tokens_generated / max(elapsed, 1e-6)

                self._send_json({
                    "question": clean_question,
                    "answer": response_text.strip(),
                    "text": response_text.strip(),
                    "tokens_generated": tokens_generated,
                    "time_ms": elapsed * 1000.0,
                    "tokens_per_sec": tok_per_sec,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
        else:
            self.send_error(404, "Endpoint not found")


def run_server(port: int = 8000, host: str = "0.0.0.0") -> None:
    # Preload model in memory
    get_model()
    server_address = (host, port)
    httpd = HTTPServer(server_address, PortfolioHTTPHandler)
    print("=" * 60)
    print(f"🚀 Karthik Jayan AI Portfolio running at: http://localhost:{port}")
    print(f"   Powered by Apple MLX (Apple Silicon M4)")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


# Backward-compatible alias
PlaygroundHTTPHandler = PortfolioHTTPHandler

