[在 Bilibili 观看项目演示视频](https://www.bilibili.com/video/BV1MFbK6zEG5/?)

[English](README.md) | [简体中文](README_zh-CN.md)

# 基于 CBF-QP 学习的机器人避碰导航算法

**A Practical Algorithm for Collision-Safe Navigation Based on CBF-QP Learning**

本项目面向 **Arduino Alvik 差速移动机器人**，研究如何将 Soft Actor-Critic（SAC）强化学习策略与控制障碍函数二次规划（CBF-QP）安全层结合，实现目标导航与避障。项目包含二维训练环境、三种算法方案的对照实验、Webots 验证，以及 Alvik 单障碍和双障碍场景的实机实验。

策略网络与安全层使用 ToF 测距、里程计及短时障碍物记忆，不接收机器人的真实位姿或障碍物的真实几何信息。仿真中的真实状态仅用于环境计算和性能评估。

> **源码发布范围：** 本仓库包含源码、配置文件、测试与说明文档，不包含预训练权重、实验日志与结果、图表、视频及二进制网格资源。执行依赖模型的评估或部署前，需要自行训练并指定模型路径。Webots 外观模型引用的 `webots_alvik/meshes/alvik_reference_design.stl` 需要另行提供。
>
> 本中文说明依据当前仓库代码整理。英文说明中的部分数值属于早期开发记录；当前主配置的障碍物记忆时长为 **8 秒**，历史 5 秒配置见 [`configs/archive/main_80cm_ttl5.json`](configs/archive/main_80cm_ttl5.json)。原英文说明引用的 `PLAN.MD` 未包含在此次源码发布中。

## 1. 三种对照方案

三种方法使用统一的机器人模型、观测空间、奖励函数、网络结构和训练预算，主要区别在于是否启用 CBF-QP，以及安全修正如何参与 SAC 更新。

设策略输出为 `u_nom`，QP 修正后的动作为 `u_safe`，安全修正量为：

```text
delta_u = u_safe - u_nom
```

| 运行模式 | 方法 | 实际执行动作 | Critic 当前动作 | Bellman 下一动作 / Actor 优化动作 | 梯度通过 QP |
|---|---|---|---|---|---|
| `vanilla` | M1：标准 SAC | `u_nom` | `u_nom` | `pi(s')` / `pi(s)` | 不使用 QP |
| `external_qp` | M2：外置 CBF-QP | `u_safe` | `u_safe` | `pi(s')` / `pi(s)` | 否 |
| `diff_qp` | M3：可微 CBF-QP | `u_safe` | `u_safe` | `QP(s', pi(s'))` / `QP(s, pi(s))` | 是 |

M2 与 M3 使用同一个 QP 求解实现。M2 的策略目标与 Bellman 下一动作不经过 QP，M3 则将修正后的动作纳入优化，并允许 Actor 的梯度经过 QP 回传。M3 不额外引入干预惩罚项，以避免给对照实验增加新的差异因素。

实现见 [`safe_alvik/sac.py`](safe_alvik/sac.py)；相关测试见 [`tests/test_sac.py`](tests/test_sac.py)。

## 2. 环境安装与检查

项目开发时使用的环境为 Python 3.9.21、NumPy 1.26.4、PyTorch 1.12.0+cu116 和 SciPy 1.13.1。依赖范围见 [`requirements.txt`](requirements.txt)。

在仓库根目录、已激活的 Python 环境中运行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t . -v
```

以下命令中的 `python` 均指安装了项目依赖的解释器。使用 `--device cuda` 前，应确保 PyTorch 能访问 CUDA；无 CUDA 时可改为 `--device cpu`，但耗时会有所不同。

测试覆盖机器人几何与坐标变换、ToF 模型、障碍物记忆、QP 求解与梯度、SAC 更新、训练流程，以及实机运行逻辑和板端安全行为。

## 3. 实验复现流程

### G0：单元测试

先执行上一节的测试命令，确认基础模块能够正常运行。

### G1：三种方法的短程检查

```bash
python run_experiments.py --config configs/smoke.json --modes vanilla external_qp diff_qp --seeds 42 --device cpu --run-root runs_smoke --name smoke --evaluate
```

### G2：无障碍导航验证

```bash
python run_experiments.py --config configs/goal_only_debug.json --modes vanilla diff_qp --seeds 42 --steps 50000 --device cuda --evaluate
```

原实验流程要求独立评估的成功率至少达到 0.80，平均最终真实目标距离不超过 0.07 米，再进入主训练。若未达到，应先检查奖励尺度、动作映射、观测归一化、折扣因子和熵系数下限。

### G3：计算耗时测试

```bash
python run_experiments.py --config configs/main_80cm.json --modes vanilla external_qp diff_qp --seeds 42 --steps 3000 --device cuda --run-root runs_bench --name timing
```

英文说明保留了开发机器上的早期耗时记录，供估计计算成本：

| 方法 | 每个训练步 | 20 万步训练 | 8 次验证 | 单次运行合计 |
|---|---:|---:|---:|---:|
| `vanilla` | 10.93 ms | 36.5 分钟 | 2.3 分钟 | 38.7 分钟 |
| `external_qp` | 13.55 ms | 45.2 分钟 | 6.1 分钟 | 51.2 分钟 |
| `diff_qp` | 18.86 ms | 62.9 分钟 | 6.1 分钟 | 69.0 分钟 |

这些记录对应当时的硬件、配置和测量方式，并非当前环境的重新测量结果。三种方法、五个随机种子顺序运行时，原估算总耗时约为 13 小时；实际耗时受硬件和验证回合长度等因素影响。

### G4：主实验

三种方法分别使用五个随机种子，每次训练 20 万步：

```bash
python run_experiments.py --config configs/main_80cm.json --modes vanilla external_qp diff_qp --seeds 42 43 44 45 46 --steps 200000 --device cuda --evaluate
```

当前 [`configs/main_80cm.json`](configs/main_80cm.json) 使用 8 秒障碍物记忆。复现历史 5 秒实验时，将配置路径替换为 `configs/archive/main_80cm_ttl5.json`，并使用独立的 `--run-root` 保存结果。

`--steps` 会同步缩放课程学习阶段、验证间隔、模型保存间隔和随机预填充等按步数定义的计划。正式对照实验应保持配置对应的训练预算一致。

### G5：结果汇总

```bash
python summarize_experiments.py --root runs --output results/main_80cm
```

若训练时指定了其他 `--run-root`，汇总时也要相应修改 `--root`。结果、日志与图表需要在本地生成，本仓库未附带原始实验数据。

### 导出策略

将 `RUN` 替换为实际训练生成的目录名：

```bash
python export_policy.py --checkpoint runs/diff_qp/RUN/checkpoints/best.pt
```

默认导出流程面向 `diff_qp`。模型检查点携带训练配置，部署时应使用与该策略一致的标定和参数。

## 4. Webots 与实机部署

### Webots 验证

[`webots_alvik/`](webots_alvik/) 包含机器人 PROTO、三种场景、控制器和评估入口，用于验证感知几何与控制软件链路。

运行前需要准备 Webots、训练得到的检查点，以及上述 STL 外观网格文件。另需将 [`runtime.ini`](webots_alvik/controllers/alvik_diff_qp/runtime.ini) 中的 `COMMAND` 修改为本机安装了项目依赖的 Python 解释器路径；仓库中保留的是开发机器路径。

```bash
python webots_alvik/run_webots_eval.py --checkpoint runs/diff_qp/RUN/checkpoints/best.pt --episodes 20 --gui
```

场景与传感器模型生成代码见 [`webots_alvik/generate_world.py`](webots_alvik/generate_world.py)。Webots 中的位姿更新采用项目运动学模型；这些实验主要验证感知与软件处理的一致性，不用于完整评估轮胎打滑或地面牵引特性。

### Alvik 实机

实机系统采用 PC 与机器人主控板分工：

- **PC：** 接收遥测，处理 ToF 与里程计，维护障碍物记忆，执行 SAC 推理、CBF-QP 和控制指令补偿。
- **Alvik 主控板：** 运行 MicroPython 程序，执行控制命令、返回传感器数据，并独立执行底层安全保护。

控制频率为 **5 Hz**。板端程序见 [`real_robot/board/main.py`](real_robot/board/main.py)，PC 入口见 [`real_robot/pc/run_diff_qp_real.py`](real_robot/pc/run_diff_qp_real.py)。

将 [`wifi_secrets_example.py`](real_robot/board/wifi_secrets_example.py) 复制为板端使用的 `wifi_secrets.py`，在本地填写 Wi-Fi 信息和 PC 地址。真实配置已列入 `.gitignore`。

先进行不发送驱动命令的检查，将示例 IP 与检查点路径替换为实际值：

```bash
python real_robot/pc/run_diff_qp_real.py --board-ip 192.168.1.50 --checkpoint runs/diff_qp/RUN/checkpoints/best.pt --layout A
```

确认测距、里程计与控制输出后，添加 `--arm` 才会向机器人发送驱动命令：

```bash
python real_robot/pc/run_diff_qp_real.py --board-ip 192.168.1.50 --checkpoint runs/diff_qp/RUN/checkpoints/best.pt --layout A --trial 1 --arm
```

实机运行只接受 `diff_qp` 检查点。系统包含通信与遥测超时制动、到达目标停止、实验步数限制，以及板端近距离保护。实机到达目标与避碰情况需要结合场地测量或视频确认，不能仅依赖存在误差的里程计。

## 5. 代码结构

| 路径 | 功能 |
|---|---|
| [`safe_alvik/config.py`](safe_alvik/config.py) | 默认配置、参数与一致性检查 |
| [`safe_alvik/geometry.py`](safe_alvik/geometry.py) | 坐标变换、运动学积分、动作映射与控制增益补偿 |
| [`safe_alvik/tof_model.py`](safe_alvik/tof_model.py) | 四分区射线测距、标定误差、三帧中值滤波与端点提取 |
| [`safe_alvik/obstacle_memory.py`](safe_alvik/obstacle_memory.py) | 障碍物端点聚类、跨帧确认、短时记忆与观测特征 |
| [`safe_alvik/cbf_qp.py`](safe_alvik/cbf_qp.py) | 基于活跃集枚举的批量可微 QP |
| [`safe_alvik/environment.py`](safe_alvik/environment.py) | 二维导航环境、场景生成和扰动 |
| [`safe_alvik/networks.py`](safe_alvik/networks.py) | Actor 与双 Critic 网络 |
| [`safe_alvik/replay.py`](safe_alvik/replay.py) | 动作、状态转移与约束的经验回放 |
| [`safe_alvik/sac.py`](safe_alvik/sac.py) | 三种 SAC 与安全层耦合方式 |
| [`safe_alvik/trainer.py`](safe_alvik/trainer.py) | 课程学习、验证与检查点保存 |
| [`safe_alvik/evaluation.py`](safe_alvik/evaluation.py) | 独立评估、指标计算与模型选择 |
| [`configs/`](configs/) | 主实验、短程检查和历史配置 |
| [`diagnostics/`](diagnostics/) | 消融实验、感知诊断与轨迹可视化工具 |
| [`real_robot/`](real_robot/) | 实机板端程序与 PC 控制程序 |
| [`webots_alvik/`](webots_alvik/) | Webots 模型、场景、控制器与评估脚本 |
| [`tests/`](tests/) | 自动化测试 |
| [`seeds5_ttl8_visualization/`](seeds5_ttl8_visualization/) | 8 秒记忆实验及记忆时长对比的绘图脚本 |
| [`seeeds_total5_visualization/`](seeeds_total5_visualization/) | 五随机种子实验的绘图脚本 |

## 6. 关键设计说明

### 可微 QP 求解

QP 对策略给出的线速度和角速度进行最小安全修正。由于只有两个决策变量，可枚举大小为 0、1 或 2 的活跃约束集合，以闭式表达计算候选解，并选取满足约束且目标值最小的候选。

该实现支持批量计算，并通过选中候选解的计算路径进行自动求导。它避免了训练期间重复调用通用迭代求解器；活跃集切换处的可微性仍需区别于固定活跃集内的情况。

### CPU 交互与 GPU 批量更新

单样本推理和环境交互放在 CPU，批量训练更新可放在 GPU。原开发环境中，单样本计算容易受 GPU 内核启动开销影响；批量计算则更适合 GPU。验证使用冻结的 CPU 策略副本，并通过测试检查动作输出一致性。

### 有限视场与障碍物记忆

ToF 分区提供距离而非精确方位，因此障碍物端点按分区中心方向估计，存在方位误差。项目通过障碍物安全余量和短时记忆处理部分感知局限。

当前默认设置为 8 厘米聚类阈值、最多 4 条轨迹、8 秒记忆时长和 2 帧确认。较短的记忆可能使离开前方视场的障碍物过早消失；较长的记忆也可能让错误测距形成的“虚假障碍物”保留更久。

以下是英文说明中记录的早期随机策略诊断结果，每组评估 200 回合，不能直接作为训练后策略的成绩：

| 设置 | 碰撞率 | 出现负净空的回合比例 | QP 不可行步比例 |
|---|---:|---:|---:|
| 不使用安全层 | 23.0% | 38.0% | — |
| QP，记忆 1.4 秒 | 10.0% | 30.5% | 3.66% |
| QP，记忆 3 秒 | 6.0% | 26.0% | 4.82% |
| QP，记忆 5 秒 | 1.5% | 16.5% | 7.80% |
| QP，记忆 8 秒 | 1.5% | 13.0% | 10.19% |

跨帧确认用于降低异常测距直接生成约束的概率，未确认轨迹不能挤占已确认轨迹。它改善了感知稳定性，但不能消除误检与漏检。

### 里程计与控制补偿

项目对实测里程计距离约 2.7% 的高估进行补偿，PC 端通过 `robot.odom_distance_gain = 1.027` 修正读数，并根据实测驱动增益补偿控制指令。这些参数随检查点配置保存，实机运行时需要保持一致。

### QP 不可行与底层保护

当 QP 不可行时，回退逻辑将线速度置零，同时允许转向，以避免无法转离障碍物。板端还有独立的近距离阻止前进和紧急制动逻辑。

板端保护没有加入三种策略的训练环境，以免掩盖标准 SAC 的碰撞行为、影响对照实验。实机日志记录板端干预，可用于分析上层安全控制未能及时处理的情况。

### 时间限制与模型选择

达到回合时间限制时保留价值自举；碰撞和到达目标按终止状态处理。模型采用依次比较的选择规则：成功率更高、碰撞率更低、最终真实目标距离更小、回报更高。

### 实验结论的适用范围

本项目中的 CBF-QP 是基于有限、带噪感知构建的工程安全措施，不构成真实环境下的无碰撞保证。Webots 与实机实验验证了部署可行性，同时也暴露了测距异常、里程计误差、有限视场和实机样本量带来的限制。
