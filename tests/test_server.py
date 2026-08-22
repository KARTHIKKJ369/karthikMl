import json
import threading
import time
import urllib.request
import pytest
from http.server import HTTPServer
from tinylm.server import PlaygroundHTTPHandler


@pytest.fixture(scope="module")
def local_server():
    port = 8765
    httpd = HTTPServer(("127.0.0.1", port), PlaygroundHTTPHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def test_server_get_index(local_server):
    req = urllib.request.urlopen(f"{local_server}/")
    assert req.status == 200
    content = req.read().decode("utf-8")
    assert "Karthik Jayan" in content


def test_server_get_models(local_server):
    req = urllib.request.urlopen(f"{local_server}/api/models")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert "models" in data
    assert len(data["models"]) >= 1


def test_server_post_generate(local_server):
    payload = {
        "prompt": "Tell me about yourself.",
        "temperature": 0.2,
        "max_new_tokens": 15,
    }
    req = urllib.request.Request(
        f"{local_server}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "answer" in data or "text" in data
        assert "tokens_generated" in data
        assert "tokens_per_sec" in data
        assert data["tokens_generated"] > 0

