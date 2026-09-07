"""UDP link to the Alvik board: receive telemetry, send commands, keep alive.

Three safety properties are enforced here rather than left to the caller:

* **Disarmed by default.** Until `arm()` is called no drive command leaves this
  process at all, so a dry run cannot move the robot however the loop above
  behaves.
* **Heartbeat.** The board brakes if it hears nothing for 300 ms, so the last
  safe command is repeated at 10 Hz between the 5 Hz policy updates. Every send
  carries a fresh, strictly increasing sequence, because the board drops any
  packet whose sequence is not greater than the last accepted one.
* **Brake on the way out.** Leaving the context manager brakes, including when
  the caller raised.
"""

import socket
import threading
import time
from typing import List, Optional

from . import protocol


class BoardLink:
    def __init__(self, board_ip: str, command_port: int = protocol.COMMAND_PORT,
                 telemetry_port: int = protocol.TELEMETRY_PORT,
                 heartbeat_hz: float = 10.0, armed: bool = False,
                 telemetry_grace_s: float = 0.4):
        self.board_ip = board_ip
        self.command_port = command_port
        self.armed = bool(armed)
        self.heartbeat_period = 1.0 / heartbeat_hz
        # The heartbeat repeats the last command so the board's own 300 ms
        # dead-man switch does not fire between 5 Hz decisions. But repeating a
        # drive command while telemetry is DEAD means driving blind, so the
        # heartbeat decays to zero once telemetry has been silent this long.
        # Without it, raising the loop's telemetry timeout would buy robustness
        # against hiccups by paying for it in blind travel.
        self.telemetry_grace_s = float(telemetry_grace_s)

        self._send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._recv_socket.bind(("", telemetry_port))
        self._recv_socket.setblocking(False)

        self._lock = threading.Lock()
        self._sequence = 0
        self._last_command = (0.0, 0.0)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.sent_commands = 0
        self.received_frames = 0
        self.dropped_frames = 0
        self.stale_heartbeats = 0
        self._last_frame_at = time.time()

    # ------------------------------------------------------------------ setup
    def arm(self) -> None:
        self.armed = True

    def __enter__(self) -> "BoardLink":
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.brake()
            time.sleep(0.05)
            self.brake()        # twice: UDP, and this is the one that matters
        finally:
            self._send_socket.close()
            self._recv_socket.close()

    # --------------------------------------------------------------- sending
    def sync_sequence(self, board_last_accepted: int) -> int:
        """Continue numbering above whatever the board last accepted.

        The board keeps `last_command_sequence` until it reboots and drops any
        packet not strictly greater than it. A second run of this script in the
        same board session would therefore start at 1, be rejected packet for
        packet, and the robot would sit in COMMAND_TIMEOUT_STOP looking exactly
        like a dead link. That is what field 18 of the telemetry is for.
        """
        with self._lock:
            self._sequence = max(self._sequence, int(board_last_accepted) + 1)
            return self._sequence

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _send(self, payload: bytes) -> None:
        try:
            self._send_socket.sendto(payload, (self.board_ip, self.command_port))
        except OSError:
            pass

    def send_command(self, linear_mps: float, angular_rps: float) -> bool:
        """Queue a drive command. Returns False when disarmed (nothing sent)."""
        with self._lock:
            self._last_command = (float(linear_mps), float(angular_rps))
            if not self.armed:
                return False
            payload = protocol.encode_command(self._next_sequence(), linear_mps, angular_rps)
            self.sent_commands += 1
        self._send(payload)
        return True

    def brake(self) -> None:
        with self._lock:
            self._last_command = (0.0, 0.0)
            if not self.armed:
                return
            payload = protocol.encode_brake(self._next_sequence())
        self._send(payload)

    def send_reset(self, pose) -> bool:
        with self._lock:
            if not self.armed:
                return False
            payload = protocol.encode_reset(self._next_sequence(), pose)
        self._send(payload)
        return True

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_period):
            with self._lock:
                if not self.armed:
                    continue
                if time.time() - self._last_frame_at > self.telemetry_grace_s:
                    # Telemetry is dead: keep the link alive but stop driving.
                    linear, angular = 0.0, 0.0
                    self.stale_heartbeats += 1
                else:
                    linear, angular = self._last_command
                payload = protocol.encode_command(self._next_sequence(), linear, angular)
            self._send(payload)

    # ------------------------------------------------------------- receiving
    def receive_frames(self, timeout_s: float) -> List[protocol.Telemetry]:
        """Every valid frame waiting on the socket, oldest first.

        The control loop runs at 5 Hz and the board reports at 10 Hz, so a
        backlog is the normal case rather than an error. The caller decides on
        the newest one and replays the rest into the board-latch mirror: the
        board evaluates its 100 mm filter on every sample it takes, so a mirror
        fed only every other sample would miss transients the board acted on.
        """
        deadline = time.time() + timeout_s
        frames: List[protocol.Telemetry] = []
        while True:
            try:
                payload, _ = self._recv_socket.recvfrom(2048)
            except BlockingIOError:
                if frames or time.time() >= deadline:
                    break
                time.sleep(0.002)
                continue
            except OSError:
                break
            frame = protocol.decode_telemetry(payload)
            if frame is None:
                self.dropped_frames += 1
                continue
            self.received_frames += 1
            self._last_frame_at = time.time()
            frames.append(frame)
        return frames

    def receive_latest(self, timeout_s: float) -> Optional[protocol.Telemetry]:
        """Newest valid frame within the timeout, discarding any backlog.

        Acting on a queued stale frame would be worse than acting on the newest
        one, so the socket is drained and only the last frame is kept.
        """
        frames = self.receive_frames(timeout_s)
        return frames[-1] if frames else None
