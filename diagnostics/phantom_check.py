

import collections
import glob
import io
import json
import math
import os
import socket
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

from real_robot.pc import protocol  # noqa: E402

try:
    sys.stdout.reconfigure(errors="replace")
except AttributeError:
    pass


def deployment_config():
    from safe_alvik.config import Config
    from safe_alvik.io_utils import load_checkpoint
    for root in ("runs_g4_ttl8", "runs_g4"):
        best, key = None, None
        for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, root, "diff_qp",
                                                  "main_seed*", "checkpoints", "best.pt"))):
            summary = os.path.join(os.path.dirname(os.path.dirname(path)),
                                   "evaluation_summary.json")
            if not os.path.exists(summary):
                continue
            with open(summary, encoding="utf-8") as handle:
                data = json.load(handle)
            candidate = (data["success"], -data["collision"])
            if key is None or candidate > key:
                best, key = path, candidate
        if best is not None:
            return Config.from_dict(load_checkpoint(best)["config"]), best
    raise SystemExit("no evaluated diff_qp checkpoint found")


def capture(seconds):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", protocol.TELEMETRY_PORT))
    sock.settimeout(3.0)
    frames, started = [], time.time()
    print("listening on UDP %d for %.0f s ..." % (protocol.TELEMETRY_PORT, seconds), flush=True)
    while time.time() - started < seconds:
        try:
            payload, _ = sock.recvfrom(2048)
        except socket.timeout:
            break
        decoded = protocol.decode_telemetry(payload)
        if decoded is not None:
            frames.append(decoded)
    sock.close()
    return frames


def replay(log_path, obstacles):
    """复查一次跑完的试验：记忆槽里有几条不对应任何真实障碍物。

    静态检查只能验证起点那一个位姿。车一开起来，位置和朝向都在变，掠射地面回波的
    几何随之改变——行进中的幻影只有靠回放日志才看得见。同时也会把地图外、但仍在
    0.70 m 端点接收范围内的房间物体算进来，它们同样占记忆槽。
    """
    import csv as _csv
    from safe_alvik.config import Config
    from safe_alvik.obstacle_memory import ObstacleMemory
    from safe_alvik.tof_model import ToFSensor

    cfg, _ = deployment_config()
    quiet = Config.from_dict(cfg.to_dict())
    quiet.randomization.enabled = False
    sensor = ToFSensor(quiet.tof, quiet.randomization,
                       quiet.robot.tof_offset_x_m, quiet.robot.tof_to_bumper_m)
    memory = ObstacleMemory(cfg.memory, cfg.tof.endpoint_accept_range_m)
    with io.open(log_path, encoding="utf-8") as handle:
        rows = list(_csv.DictReader(handle))
    truth = (np.asarray(obstacles, dtype=float).reshape(-1, 3)
             if obstacles else np.zeros((0, 3)))

    # 端点只报距离不报方位，落点在区中心线上，误差上限 0.131*d；再加一个聚类半径。
    tolerance = 0.131 * cfg.tof.endpoint_accept_range_m + cfg.memory.cluster_radius_m
    now, slots, spurious, worst, far = 0.0, [], [], 0, []
    for row in rows:
        now += cfg.robot.control_dt_s
        pose = np.array([float(row["odom_x"]), float(row["odom_y"]),
                         math.radians(float(row["odom_theta_deg"]))])
        channels = np.array([float(row[c]) for c in
                             ("L_mm", "CL_mm", "C_mm", "CR_mm", "R_mm")]) / 1000.0
        memory.update(sensor.endpoints(pose, sensor.push(channels)), now)
        points = memory.points()
        slots.append(len(points))
        count = 0
        for point in points:
            if len(truth):
                gap = float(np.min(np.linalg.norm(point - truth[:, :2], axis=1) - truth[:, 2]))
            else:
                gap = 9.9
            if gap > tolerance:
                count += 1
                far.append(point.copy())
        spurious.append(count)
        worst = max(worst, count)

    spurious = np.array(spurious)
    slots = np.array(slots)
    print()
    print("回放 %s" % os.path.basename(log_path))
    print("已知障碍物 %d 个；判定容差 %.3f m = 方位误差 %.3f + 聚类 %.2f"
          % (len(truth), tolerance, 0.131 * cfg.tof.endpoint_accept_range_m,
             cfg.memory.cluster_radius_m))
    print("记忆槽上限 %d 条" % cfg.memory.max_tracks)
    print()
    print("平均占用 %.2f 条，其中不对应任何已知障碍物的平均 %.2f 条，最多 %d 条"
          % (slots.mean(), spurious.mean(), worst))
    print("有杂散轨迹的步数占 %.1f%%" % (100.0 * (spurious > 0).mean()))
    if far:
        cluster = []
        for point in far:
            if not cluster or min(np.linalg.norm(point - np.array(cluster), axis=1)) > 0.15:
                cluster.append(point)
        print("杂散轨迹大致位置（里程计坐标）：")
        for point in cluster[:6]:
            print("   (%+.3f, %+.3f)" % (point[0], point[1]))
    print()
    if spurious.mean() < 0.5:
        print("干净。记忆基本都在跟真实障碍物。")
        return 0
    print("杂散轨迹会挤占记忆槽（上限 %d），也会进 QP 约束和四象限观测特征。"
          % cfg.memory.max_tracks)
    print("实测把这类杂物搬进仿真后，同一摆位的穿缝率从 24/25 掉到 0/25——不是小噪声。")
    print()
    print("两个来源都要处理：")
    print("  * 地图外的房间物体：端点接收范围 %.2f m，地图半宽只有 %.2f m，"
          % (cfg.tof.endpoint_accept_range_m, cfg.map.half_extent_m))
    print("    所以地图外 %.2f m 之内的任何东西都会被记进来。清场按前者算，不是按地图边界。"
          % (cfg.tof.endpoint_accept_range_m - cfg.map.half_extent_m))
    print("  * 行进中的地面回波：静态检查只验证起点那一个位姿，车开起来朝向一变，")
    print("    掠射几何跟着变，只有这个回放模式看得见。")
    return 1


def main(argv):
    if "--log" in argv:
        i = argv.index("--log")
        obstacles = []
        while "--obstacle" in argv:
            k = argv.index("--obstacle")
            obstacles.append([float(v) for v in argv[k + 1:k + 4]])
            del argv[k:k + 4]
        return replay(argv[i + 1], obstacles)
    seconds = float(argv[1]) if len(argv) > 1 else 60.0
    frames = capture(seconds)
    if len(frames) < 10:
        print("遥测太少（%d 帧），先用 tof_live_check.py 确认链路通。" % len(frames))
        return 2

    cfg, checkpoint = deployment_config()
    from safe_alvik.cbf_qp import build_constraint_rows
    from safe_alvik.config import Config
    from safe_alvik.obstacle_memory import ObstacleMemory
    from safe_alvik.tof_model import ToFSensor

    quiet = Config.from_dict(cfg.to_dict())
    quiet.randomization.enabled = False          # 真传感器，不叠仿真误差模型
    sensor = ToFSensor(quiet.tof, quiet.randomization,
                       quiet.robot.tof_offset_x_m, quiet.robot.tof_to_bumper_m)
    memory = ObstacleMemory(cfg.memory, cfg.tof.endpoint_accept_range_m)
    pose = np.array(cfg.map.start_pose, dtype=np.float64)   # 静止：位姿固定即可

    # 板端 10 Hz、策略 5 Hz，所以每两帧决策一次，和 run_diff_qp_real 一致。
    dt = cfg.robot.control_dt_s
    constrained_steps, steps, now = 0, 0, 0.0
    track_counts, worst = collections.Counter(), []
    for index in range(0, len(frames) - 1, 2):
        now += dt
        steps += 1
        channels = sensor.push(frames[index + 1].channels)
        memory.update(sensor.endpoints(pose, channels), now)
        points = memory.points()
        rows = build_constraint_rows(pose, points, cfg.robot, cfg.cbf, cfg.memory.max_tracks)
        active = int(rows[2].sum())
        track_counts[active] += 1
        if active:
            constrained_steps += 1
            distances = np.linalg.norm(points - pose[:2], axis=1)
            worst.append(float(distances.min()))

    print()
    print("checkpoint  %s" % os.path.relpath(checkpoint, PROJECT_ROOT))
    print("参数        ttl %.1f s, confirm %d 帧, 中值 %d 帧, 端点接收 %.2f m, 聚类 %.2f m"
          % (cfg.memory.ttl_s, cfg.memory.confirm_frames, cfg.tof.median_frames,
             cfg.tof.endpoint_accept_range_m, cfg.memory.cluster_radius_m
             if hasattr(cfg.memory, "cluster_radius_m") else 0.08))
    print("样本        %d 帧遥测 -> %d 个控制步（%.1f s）" % (len(frames), steps, steps * dt))
    print()
    fraction = 100.0 * constrained_steps / max(steps, 1)
    print("**cons > 0 的步数：%d / %d = %.1f%%**" % (constrained_steps, steps, fraction))
    print("约束条数分布：%s" % dict(sorted(track_counts.items())))
    if worst:
        print("幻影最近距离：中位 %.3f m，最近 %.3f m（安全半径 %.3f m）"
              % (float(np.median(worst)), float(np.min(worst)), cfg.robot.safety_radius_m))
    print()
    if constrained_steps == 0:
        print("干净。空地上不会产生幻影约束，可以进空跑核对那一步。")
        return 0
    print("""不干净。武装之后 CBF 会去躲这些不存在的点，表现为无故减速或偏航。

先回到 tof_live_check.py 把偏低的那个通道压到 0.0%%，再回来复测这一项。
中值滤波能杀掉孤立单帧，但连续两帧就足以凑够 confirm_frames=%d 的确认，
而 TTL 是 %.0f s——也就是说 %.0f 秒里出现一次连续两帧的假低值，就足以让
一条幻影轨迹几乎全程在线。""" % (cfg.memory.confirm_frames, cfg.memory.ttl_s, cfg.memory.ttl_s))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
