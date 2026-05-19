from __future__ import annotations

import json
import socket
import time


HOST = "127.0.0.1"
PORT = 8765


EXAMPLE_EVENTS = [
    {
        "type": "speech",
        "text": "안녕",
        "timestamp": time.time(),
    },
    {
        "type": "gesture",
        "name": "wave",
        "value": "HELLO",
        "timestamp": time.time(),
    },
    {
        "type": "emotion",
        "label": "Happy",
        "scores": {
            "Happy": 0.72,
            "Neutral": 0.18,
            "Sadness": 0.04,
        },
        "timestamp": time.time(),
    },
]


def main() -> None:
    with socket.create_connection((HOST, PORT), timeout=5) as client:
        for event in EXAMPLE_EVENTS:
            payload = json.dumps(event, ensure_ascii=False) + "\n"
            client.sendall(payload.encode("utf-8"))
            print(f"sent: {event['type']}")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
