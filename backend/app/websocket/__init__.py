"""
SETU — WebSocket Routing Package
Low-latency real-time video streaming, backpressure management, and recognition handlers.
"""

from __future__ import annotations

from backend.app.websocket.recognize import websocket_recognize

__all__ = [
    "websocket_recognize",
]
