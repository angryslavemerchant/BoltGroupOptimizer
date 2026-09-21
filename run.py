import threading
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    uvicorn.run("app.main:app", host=HOST, port=PORT)
