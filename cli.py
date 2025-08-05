import uvicorn

from .api.app import app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    url = f"http://{host}:{port}"

    print(f"🚀 RePViz dashboard is running at: {url}")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
