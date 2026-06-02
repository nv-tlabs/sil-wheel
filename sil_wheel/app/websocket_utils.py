# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json

import websockets
from websockets.server import WebSocketServerProtocol

WS_HOST = "0.0.0.0"
WS_PORT = 7000

connected_clients: "set[WebSocketServerProtocol]" = set()
websocket_loop = None


async def _ws_handler(websocket: WebSocketServerProtocol, *args):
    print(
        f"[ws] client connected: {getattr(websocket, 'remote_address', None)}"
    )
    connected_clients.add(websocket)
    global OPTIONS

    try:
        # Send welcome message
        welcome_msg = {
            "type": "welcome",
            "message": "Connected to server",
            "client_count": len(connected_clients),
        }
        await websocket.send(json.dumps(welcome_msg))

        global OPTIONS

        # Keep connection alive
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "set_options":
                    OPTIONS = data.get("options")
                    assert isinstance(OPTIONS, list), "Options must be a list"
                    for option in OPTIONS:
                        print(f"- {option}")
                else:
                    print(f"[ws] Received: {data}")
            except json.JSONDecodeError:
                print(f"[ws] eceived (raw): {message}")

    except websockets.exceptions.ConnectionClosed:
        print("[ws] WebSocket client disconnected")
    except Exception as e:
        print(f"[ws] handler error: {e}")
    finally:
        connected_clients.discard(websocket)
        print(f"[ws] Total connected clients: {len(connected_clients)}")


async def _ws_broadcast(message: str):
    if not connected_clients:
        return
    stale = []
    for ws in list(connected_clients):
        try:
            await ws.send(message)
        except Exception:
            stale.append(ws)
    for ws in stale:
        connected_clients.discard(ws)
    if stale:
        print(f"[ws] cleaned up {len(stale)} stale clients")


def ws_broadcast_threadsafe(obj):
    """Call from any thread to broadcast a JSON-serializable payload."""
    global websocket_loop
    if websocket_loop is None:
        print("[ws] server not ready; no broadcast")
        return
    fut = asyncio.run_coroutine_threadsafe(
        _ws_broadcast(json.dumps(obj)), websocket_loop
    )
    try:
        fut.result(timeout=0.0)  # fire-and-forget
    except Exception:
        pass


async def _ws_main():
    """Runs inside the WS thread; sets running loop and serves forever."""
    global websocket_loop
    websocket_loop = asyncio.get_running_loop()
    async with websockets.serve(_ws_handler, WS_HOST, WS_PORT):
        print(f"[ws] listening on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()  # run forever


def run_ws_server():
    """Thread target: spin up an event loop and run the websocket server."""
    asyncio.run(_ws_main())
