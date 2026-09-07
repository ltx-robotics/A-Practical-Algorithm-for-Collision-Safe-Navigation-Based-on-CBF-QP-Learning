

import collections
import os
import socket
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

from real_robot.pc import protocol  # noqa: E402
from real_robot.pc.board_safety import (CRITICAL_STOP_FRONT_MM, HARD_STOP_FRONT_MM,  # noqa: E402
                                        CONSECUTIVE_FRAMES)

# Windows 控制台的代码页不一定能编码这里用到的符号，宁可显示成 ? 也不要在
# 机器旁边抛 UnicodeEncodeError 把整份报告弄丢。
try:
    sys.stdout.reconfigure(errors="replace")
except AttributeError:
    pass

ZONES = ["L", "CL", "C", "CR", "R"]
BEARING = {"L": "+22.5°(左)", "CL": "+7.5°", "C": "合成", "CR": "-7.5°", "R": "-22.5°(右)"}
LOW_MM = 200.0


def capture(seconds):
    import time
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", protocol.TELEMETRY_PORT))
    sock.settimeout(3.0)
    frames, started = [], time.time()
    print("listening on UDP %d for %.0f s ..." % (protocol.TELEMETRY_PORT, seconds),
          flush=True)
    while time.time() - started < seconds:
        try:
            payload, _ = sock.recvfrom(2048)
        except socket.timeout:
            break
        decoded = protocol.decode_telemetry(payload)
        if decoded is not None:
            frames.append(decoded)
    sock.close()
    return frames, time.time() - started


def longest_run(flags):
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def live():
    """逐帧刷新，用于一边动手一边看是哪个通道在响应。"""
    import time
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", protocol.TELEMETRY_PORT))
    sock.settimeout(3.0)
    print("实时 ToF（mm）。在某个方位晃手，看哪一列跟着动。Ctrl-C 停。")
    print()
    print("      L     CL      C     CR      R    板端状态")
    try:
        while True:
            try:
                payload, _ = sock.recvfrom(2048)
            except socket.timeout:
                print("  ...没有遥测")
                continue
            frame = protocol.decode_telemetry(payload)
            if frame is None:
                continue
            mm = frame.channels * 1000.0
            marks = "".join("<" if v < LOW_MM else " " for v in mm)
            print("  %5.0f  %5.0f  %5.0f  %5.0f  %5.0f   %s  %s"
                  % (mm[0], mm[1], mm[2], mm[3], mm[4], marks, frame.state), flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
        print("停止。")
    finally:
        sock.close()
    return 0


def main(argv):
    if len(argv) > 1 and argv[1].lower() in ("live", "watch", "-l"):
        return live()
    seconds = float(argv[1]) if len(argv) > 1 else 30.0
    frames, elapsed = capture(seconds)
    if not frames:
        print("没收到遥测。检查：板端在跑 main.py、wifi_secrets.py 里的 PC_IP 是这台电脑、"
              "防火墙放行 UDP %d、两边同一网段。" % protocol.TELEMETRY_PORT)
        return 2

    channels = np.array([f.channels for f in frames]) * 1000.0
    print("\n收到 %d 帧 / %.1f s（%.1f Hz）\n" % (len(frames), elapsed, len(frames) / elapsed))
    print("%-4s %-12s %7s %7s %7s %9s %9s" %
          ("通道", "方位", "最小", "中位", "最大", "<200mm", "最长连续低"))
    suspicious = []
    for index, name in enumerate(ZONES):
        column = channels[:, index]
        low = column < LOW_MM
        run = longest_run(low)
        print("%-4s %-12s %7.0f %7.0f %7.0f %8.1f%% %9d"
              % (name, BEARING[name], column.min(), np.median(column), column.max(),
                 100.0 * low.mean(), run))
        if low.mean() > 0.05:
            suspicious.append((name, low.mean(), np.median(column[low]), run))

    print("\n板端安全层（只看 CL/C/CR）：")
    front = channels[:, [1, 2, 3]]
    blocked = longest_run((front <= HARD_STOP_FRONT_MM).any(axis=1))
    critical = int((front <= CRITICAL_STOP_FRONT_MM).any(axis=1).sum())
    print("  ≤%.0f mm 的最长连续帧数 %d（≥%d 就会禁止前进）"
          % (HARD_STOP_FRONT_MM, blocked, CONSECUTIVE_FRAMES))
    print("  ≤%.0f mm 的帧数 %d（任意一帧就会锁存急停）" % (CRITICAL_STOP_FRONT_MM, critical))

    states = collections.Counter(f.state for f in frames)
    print("\nboard_state：%s" % dict(states))
    if set(states) == {"COMMAND_TIMEOUT_STOP"}:
        print("  （没人发命令时就是这个状态，空跑时属于正常）")

    print("\n结论：")
    if not suspicious:
        print("  五个通道都干净，可以进下一步。")
        return 0
    for name, fraction, median_low, run in suspicious:
        kind = "长期卡住" if fraction > 0.8 else "间歇性"
        print("  ! %s 通道%s偏低：%.0f%% 的帧读到约 %.0f mm，最长连续 %d 帧"
              % (name, kind, 100.0 * fraction, median_low, run))
    print("""
  这些读数会在里程计坐标里造出幻影障碍物，CBF 会一直躲它。武装之后的表现是
  全程龟速或原地打转，日志里 `n_constraints` 恒 >0 而地面上什么都没有。

  按顺序排查：
    1. 看机器人正前方 20 cm 内有没有实物——线缆、夹爪、胶带、地图翘起的边。
    2. 用软布擦 ToF 窗口。窗口上的指纹或灰会让特定分区一直读到十几厘米。
    3. 对照实验：把机器人举起来对着空旷方向（2 m 内无物），再跑一次本脚本。
       读数变干净 = 环境问题；还是低 = 传感器或遮挡问题。
    4. 相邻两区互相印证：L 很近而 CL 很远（或 CR 近而 R 远）不可能是真障碍物，
       那是传感器问题，不是环境问题。""")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
