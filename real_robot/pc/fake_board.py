"""Loopback stand-in for the Alvik board, for smoke-testing the PC runtime.

    python real_robot/pc/fake_board.py 75      # in one terminal
    python real_robot/pc/run_diff_qp_real.py --board-ip 127.0.0.1 --arm


Integrates the commands it receives with the measured gains, ray-casts the five
ToF zones against a fake obstacle, and streams telemetry at 10 Hz - enough to
prove the CLI's whole path without hardware. Not a substitute for the trained
simulator; it exists to catch a broken socket, a wrong port or a crash on step 1.
"""

import math
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from safe_alvik.config import Config
from safe_alvik.geometry import unicycle_step
from safe_alvik.tof_model import ideal_zone_distances, to_channels

COMMAND_PORT = 4212
TELEMETRY_PORT = 4211
DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
OBSTACLES = np.array([[0.0, 0.0, 0.05]])

cfg = Config()
pose = np.array(cfg.map.start_pose, dtype=np.float64)
command = np.zeros(2)
last_command_ms = None
last_sequence = -1
sequence = 0

recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
recv.bind(("127.0.0.1", COMMAND_PORT))
recv.setblocking(False)
send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("FAKE_BOARD_READY", flush=True)
started = time.time()
last_telemetry = 0.0
accepted = 0

while time.time() - started < DURATION_S:
    now_ms = int((time.time() - started) * 1000)
    while True:
        try:
            payload, _ = recv.recvfrom(256)
        except BlockingIOError:
            break
        fields = payload.decode().strip().split(",")
        seq = int(fields[1])
        if seq <= last_sequence:
            continue
        last_sequence = seq
        accepted += 1
        if fields[0] == "CMD":
            command = np.array([float(fields[2]) / 100.0, math.radians(float(fields[3]))])
            last_command_ms = now_ms
        elif fields[0] == "BRAKE":
            command = np.zeros(2)
            last_command_ms = now_ms
        elif fields[0] == "RESET":
            pose = np.array([float(fields[2]) / 1000.0, float(fields[3]) / 1000.0,
                             math.radians(float(fields[4]))])
            command = np.zeros(2)
            last_command_ms = now_ms

    stale = last_command_ms is None or (now_ms - last_command_ms) > 300
    applied = np.zeros(2) if stale else command * [cfg.robot.gain_linear,
                                                   cfg.robot.gain_angular_ccw]
    pose = unicycle_step(pose, applied[0], applied[1], 0.01)

    if time.time() - started - last_telemetry >= 0.1:
        last_telemetry = time.time() - started
        sequence += 1
        zones = ideal_zone_distances(pose, OBSTACLES, cfg.tof, cfg.robot.tof_offset_x_m)
        channels = to_channels(zones) * 1000.0
        # The board reports its own uncorrected odometry, which over-reads.
        reported = pose.copy()
        reported[:2] = cfg.map.start_pose[:2] + (pose[:2] - cfg.map.start_pose[:2]) * \
            cfg.robot.odom_distance_gain
        fields = ["TEL", str(sequence), str(now_ms),
                  "%.3f" % (reported[0] * 1000.0), "%.3f" % (reported[1] * 1000.0),
                  "%.4f" % math.degrees(reported[2])]
        fields += ["%.2f" % value for value in channels]
        fields.append("%.2f" % (channels[2] - 8.0))
        fields += ["%.4f" % (command[0] * 100.0), "%.4f" % math.degrees(command[1])]
        fields += ["%.4f" % (applied[0] * 100.0), "%.4f" % math.degrees(applied[1])]
        fields += ["COMMAND_ACTIVE", str(last_sequence)]
        send.sendto(",".join(fields).encode(), ("127.0.0.1", TELEMETRY_PORT))
    time.sleep(0.01)

print("FAKE_BOARD_DONE accepted=%d telemetry=%d final=(%.3f,%.3f)"
      % (accepted, sequence, pose[0], pose[1]), flush=True)
