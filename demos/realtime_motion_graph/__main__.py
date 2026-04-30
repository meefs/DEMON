"""Local-dev launcher: run the server and client together from one command.

Usage:
    uv run python -m demos.realtime_motion_graph --audio song.wav
    uv run python -m demos.realtime_motion_graph --midi --audio song.wav

Spawns the server as a subprocess on 127.0.0.1, waits for it to print
its "Listening..." line, then runs the client in the foreground with
``--remote ws://127.0.0.1:<port>``. Ctrl-C in the client shuts both
down.

Client flags (``--midi``, ``--sde``, ``--lora``, ``--audio``, etc.) pass
through unchanged. ``--host`` and ``--port`` configure the server.
"""

import subprocess
import sys
import threading
import time


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "8765"
READY_TIMEOUT_SECONDS = 180


def _pop_value(args, name, default=None):
    if name in args:
        i = args.index(name)
        val = args[i + 1]
        del args[i:i + 2]
        return val
    return default


def main():
    args = list(sys.argv[1:])
    host = _pop_value(args, "--host", DEFAULT_HOST)
    port = str(_pop_value(args, "--port", DEFAULT_PORT))

    server_cmd = [
        sys.executable, "-u", "-m", "demos.realtime_motion_graph.server",
        "--host", host, "--port", port,
    ]
    print(f"[launcher] Starting server on {host}:{port}")
    srv = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    ready = threading.Event()
    server_exited = threading.Event()

    def _pipe_server_output():
        try:
            for line in srv.stdout:
                line = line.rstrip()
                if line:
                    print(f"[server] {line}")
                if "Listening" in line:
                    ready.set()
        finally:
            server_exited.set()
            ready.set()  # unblock waiter if the server died before becoming ready

    pipe_thread = threading.Thread(target=_pipe_server_output, daemon=True)
    pipe_thread.start()

    print(f"[launcher] Waiting for server to be ready (timeout {READY_TIMEOUT_SECONDS}s)...")
    if not ready.wait(timeout=READY_TIMEOUT_SECONDS):
        print("[launcher] Server did not become ready in time, aborting")
        srv.terminate()
        sys.exit(1)
    if server_exited.is_set() and srv.poll() is not None:
        print(f"[launcher] Server exited before client could connect (rc={srv.returncode})")
        sys.exit(srv.returncode or 1)

    client_cmd = [
        sys.executable, "-m", "demos.realtime_motion_graph.client",
        "--remote", f"ws://{host}:{port}",
        *args,
    ]
    print(f"[launcher] Starting client: {' '.join(client_cmd[2:])}")
    rc = 0
    try:
        rc = subprocess.call(client_cmd)
    except KeyboardInterrupt:
        rc = 0
    finally:
        print("[launcher] Client exited, stopping server...")
        if srv.poll() is None:
            srv.terminate()
            try:
                srv.wait(timeout=5)
            except subprocess.TimeoutExpired:
                srv.kill()
                srv.wait(timeout=2)
        pipe_thread.join(timeout=1)
        print(f"[launcher] Done (client rc={rc})")

    sys.exit(rc)


if __name__ == "__main__":
    main()
