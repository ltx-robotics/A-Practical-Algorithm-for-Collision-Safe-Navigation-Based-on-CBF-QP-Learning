

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safe_alvik.config import Config
from safe_alvik.tof_model import zone_ray_angles

HERE = os.path.dirname(os.path.abspath(__file__))
ZONE_NAMES = ("L", "CL", "CR", "R")

# Measured on the real robot (calibration_and_measurement/Alvik_calibration_note):
# effective wheel radius 17 mm, geometric track width 90 mm. The previous
# project's PROTO used a 34 mm radius, which put the chassis - and therefore the
# ToF beam - twice as high above the ground.
WHEEL_RADIUS_M = 0.017
TRACK_WIDTH_M = 0.090
CHASSIS_THICKNESS_M = 0.028
# Beam height above the floor. Not separately measured; obstacles are required
# to be at least 0.12 m tall (PLAN.MD 5.1) so detection is insensitive to it.
TOF_HEIGHT_M = WHEEL_RADIUS_M + 0.5 * CHASSIS_THICKNESS_M

OBSTACLE_HEIGHT_M = 0.12

# Visual body: the official Alvik reference-design mesh (AKX00066-step). The
# STL is in millimetres, so it is scaled by 1e-3. Its own frame was worked out
# from the geometry rather than assumed:
#   * the lowest points cluster at x = +/-43.9 mm, i.e. the two wheels, so the
#     wheel axle runs along the mesh X axis - a 87.8 mm track against the 90 mm
#     measured on the robot, which is a good independent check;
#   * the caster sits at y ~ -41 mm, so the mesh faces +Y while this project
#     drives along +X, hence the -90 deg yaw;
#   * ground contact is at z = -10.10 mm, so with a 17 mm effective wheel radius
#     the axle - the pose reference point for this whole project - sits at
#     z = +6.90 mm inside the mesh.
MESH_FILE = "../meshes/alvik_reference_design.stl"
MESH_SCALE = 0.001
MESH_YAW_RAD = -math.pi / 2.0
MESH_GROUND_CONTACT_MM = -10.10

# Camera framing. Webots' viewpoint camera looks along its local +X, with +Y to
# the left and +Z up - the robotics convention, not the OpenGL -Z one. This was
# established by solving the shipped sample world
# (projects/samples/devices/worlds/distance_sensor.wbt) for which local axis its
# rotation maps onto the direction from camera to scene: +X matches to within
# 2.7 deg, every other axis is 87-93 deg off. Getting this backwards aims the
# camera at the sky and the whole scene renders blank.
CAMERA_POSITION = (0.95, -0.95, 0.85)
CAMERA_TARGET = (0.0, 0.0, 0.02)
CAMERA_FOV = 0.85


def look_at(position, target, world_up=(0.0, 0.0, 1.0)):
    """Axis-angle for a camera at `position` aimed at `target`."""
    import numpy as np
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(world_up, dtype=float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    upward = np.cross(forward, left)
    matrix = np.column_stack([forward, left, upward])   # local +X/+Y/+Z -> world
    angle = float(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)))
    if abs(angle) < 1e-9:
        return (0.0, 0.0, 1.0, 0.0)
    axis = np.array([matrix[2, 1] - matrix[1, 2],
                     matrix[0, 2] - matrix[2, 0],
                     matrix[1, 0] - matrix[0, 1]]) / (2.0 * np.sin(angle))
    return (float(axis[0]), float(axis[1]), float(axis[2]), angle)


# Fixed layouts, identical to the real-robot layouts in PLAN.MD 10.3.
#
# C was originally specified as (-0.06, 0.02) and (0.10, -0.10), described as a
# ">= 0.20 m corridor". Those coordinates leave only 0.100 m between the two
# surfaces - less than the 0.132 m body diameter - so the robot could not pass
# between them at all and had to go around the pair. The corrected pair sits
# symmetrically across the start-goal line and leaves 0.260 m, i.e. 34 mm of
# spare on each side of the 0.192 m the body plus CBF margin needs. That is
# genuinely narrow but wider than the known +/- 7.5 deg endpoint bearing error
# (about 33 mm at the 0.25 m range where the QP intervenes).
LAYOUTS = {
    "A": [(0.00, 0.00, 0.05)],
    "B": [(-0.05, 0.06, 0.06)],
    "C": [(-0.127, 0.127, 0.05), (0.127, -0.127, 0.05)],
}
LAYOUT_NOTE = {
    "A": "single obstacle centred on the start-goal line",
    "B": "single obstacle offset to one side of the line",
    "C": "two obstacles leaving a 0.260 m corridor across the start-goal line",
}


def tof_sensor_nodes(cfg: Config) -> str:
    """One DistanceSensor per modelled ray, named tof_<zone>_<index>."""
    angles = zone_ray_angles(cfg.tof)          # (4, rays) radians, robot frame
    lines = []
    for zone_index, zone in enumerate(ZONE_NAMES):
        for ray_index in range(angles.shape[1]):
            angle = float(angles[zone_index, ray_index])
            lines.append(
                "      DistanceSensor {\n"
                "        translation %.6f 0 %.6f\n"
                "        rotation 0 0 1 %.6f\n"
                '        name "tof_%s_%d"\n'
                "        lookupTable [ 0 0 0, %.4f %.4f 0 ]\n"
                '        type "generic"\n'
                "        numberOfRays 1\n"
                "        aperture 0.002\n"
                "      }" % (cfg.robot.tof_offset_x_m, TOF_HEIGHT_M, angle,
                             zone, ray_index, cfg.tof.max_range_m, cfg.tof.max_range_m))
    return "\n".join(lines)


def build_proto(cfg: Config) -> str:
    half_track = 0.5 * TRACK_WIDTH_M
    return """#VRML_SIM R2025a utf8

# GENERATED BY webots_alvik/generate_world.py - DO NOT EDIT BY HAND.
# Geometry is derived from safe_alvik/config.py so it cannot drift away from the
# geometry the policy was trained with. Re-run the generator after config changes.
#
# Measured values: effective wheel radius %.3f m, geometric track width %.3f m,
# body circumscribed radius %.3f m, ToF window %.3f m ahead of the wheel axle
# midpoint. The %d ToF DistanceSensors reproduce the %d zones x %d rays of the
# training sensor model, ray for ray.
#
# The robot is a Supervisor so the controller can reset episodes and read the
# ground-truth pose for collision and success judging. The policy itself never
# sees that pose.

PROTO AlvikTof [
  field SFVec3f    translation  %.3f %.3f %.6f
  field SFRotation rotation     0 0 1 0
  field SFString   name         "Alvik"
  field SFString   controller   "alvik_diff_qp"
  field MFString   controllerArgs []
]
{
  Robot {
    translation IS translation
    rotation IS rotation
    name IS name
    controller IS controller
    controllerArgs IS controllerArgs
    supervisor TRUE
    children [
      Transform {
        translation 0 0 %.6f
        rotation 0 0 1 %.6f
        scale %.4f %.4f %.4f
        children [
          Shape {
            appearance PBRAppearance {
              baseColor 0.055 0.510 0.525
              roughness 0.42
              metalness 0.15
            }
            geometry Mesh { url [ "%s" ] }
            castShadows FALSE
          }
        ]
      }
      Pose {
        translation %.3f 0 %.6f
        children [
          Shape {
            appearance PBRAppearance { baseColor 0.98 0.62 0.09 emissiveColor 0.35 0.20 0.0 roughness 0.35 }
            geometry Box { size 0.004 0.026 0.008 }
          }
        ]
      }
%s
      HingeJoint {
        jointParameters HingeJointParameters { anchor 0 %.4f 0 axis 0 1 0 }
        device [
          RotationalMotor { name "left wheel motor" maxVelocity 40 }
          PositionSensor { name "left wheel sensor" }
        ]
        endPoint Solid {
          translation 0 %.4f 0
          rotation 1 0 0 1.5707963267948966
          name "left wheel"
          children []
        }
      }
      HingeJoint {
        jointParameters HingeJointParameters { anchor 0 %.4f 0 axis 0 1 0 }
        device [
          RotationalMotor { name "right wheel motor" maxVelocity 40 }
          PositionSensor { name "right wheel sensor" }
        ]
        endPoint Solid {
          translation 0 %.4f 0
          rotation 1 0 0 1.5707963267948966
          name "right wheel"
          children []
        }
      }
    ]
  }
}
""" % (WHEEL_RADIUS_M, TRACK_WIDTH_M, cfg.robot.body_radius_m, cfg.robot.tof_offset_x_m,
       4 * cfg.tof.rays_per_zone, 4, cfg.tof.rays_per_zone,
       cfg.map.start_pose[0], cfg.map.start_pose[1], WHEEL_RADIUS_M,
       # mesh: lift it so its ground contact lands on the floor
       -WHEEL_RADIUS_M - MESH_SCALE * MESH_GROUND_CONTACT_MM, MESH_YAW_RAD,
       MESH_SCALE, MESH_SCALE, MESH_SCALE, MESH_FILE,
       # ToF marker
       cfg.robot.tof_offset_x_m, TOF_HEIGHT_M,
       tof_sensor_nodes(cfg),
       half_track, half_track,
       -half_track, -half_track)


def obstacle_node(index: int, x: float, y: float, radius: float) -> str:
    return """Solid {
  translation %.4f %.4f %.4f
  name "obstacle_%d"
  children [
    Shape {
      appearance PBRAppearance { baseColor 0.83 0.31 0.24 roughness 0.55 metalness 0.05 }
      geometry Cylinder { height %.3f radius %.4f subdivision 32 }
    }
  ]
  boundingObject Cylinder { height %.3f radius %.4f }
}""" % (x, y, 0.5 * OBSTACLE_HEIGHT_M, index, OBSTACLE_HEIGHT_M, radius,
        OBSTACLE_HEIGHT_M, radius)


def build_world(cfg: Config, layout: str) -> str:
    obstacles = LAYOUTS[layout]
    side = 2.0 * cfg.map.half_extent_m
    goal = cfg.map.goal_position
    start = cfg.map.start_pose
    nodes = "\n".join(obstacle_node(i, x, y, r) for i, (x, y, r) in enumerate(obstacles))
    camera = look_at(CAMERA_POSITION, CAMERA_TARGET)
    return """#VRML_SIM R2025a utf8

# GENERATED BY webots_alvik/generate_world.py - DO NOT EDIT BY HAND.
# Layout %s: %s
#
# The printed map has NO physical walls and the ToF cannot see its border, so
# this world deliberately contains no boundary geometry either. The floor plate
# is only a visual reference; nothing stops the robot leaving it, exactly as in
# the Python environment and on the real robot.

EXTERNPROTO "../protos/AlvikTof.proto"

WorldInfo {
  basicTimeStep 8
  coordinateSystem "ENU"
  info [ "Alvik SAC + differentiable CBF-QP, layout %s" ]
}
Viewpoint {
  orientation %.6f %.6f %.6f %.6f
  position %.3f %.3f %.3f
  fieldOfView %.3f
}
Background { skyColor [ 0.90 0.93 0.97 ] }
DirectionalLight { direction 0.45 0.55 -1 intensity 2.2 castShadows TRUE }
DirectionalLight { direction -0.6 -0.3 -0.5 intensity 0.8 castShadows FALSE }

Solid {
  translation 0 0 -0.005
  name "printed_map"
  children [
    Shape {
      appearance PBRAppearance { baseColor 0.94 0.94 0.92 roughness 0.95 metalness 0 }
      geometry Box { size %.3f %.3f 0.01 }
    }
  ]
  boundingObject Box { size %.3f %.3f 0.01 }
}

Pose {
  translation %.3f %.3f 0.002
  children [
    Shape {
      appearance PBRAppearance { baseColor 0.18 0.70 0.38 emissiveColor 0.05 0.22 0.11 roughness 0.7 }
      geometry Cylinder { height 0.004 radius %.3f }
    }
  ]
}

%s

AlvikTof {
  translation %.3f %.3f %.6f
  rotation 0 0 1 %.6f
  controllerArgs [ "--layout" "%s" ]
}
""" % (layout, LAYOUT_NOTE[layout], layout,
       camera[0], camera[1], camera[2], camera[3],
       CAMERA_POSITION[0], CAMERA_POSITION[1], CAMERA_POSITION[2], CAMERA_FOV,
       side, side, side, side,
       goal[0], goal[1], cfg.map.success_radius_m,
       nodes, start[0], start[1], WHEEL_RADIUS_M, start[2], layout)


def main() -> int:
    cfg = Config.from_json(os.path.join(os.path.dirname(HERE), "configs", "main_80cm.json"))
    os.makedirs(os.path.join(HERE, "protos"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "worlds"), exist_ok=True)

    proto_path = os.path.join(HERE, "protos", "AlvikTof.proto")
    with open(proto_path, "w", encoding="utf-8") as handle:
        handle.write(build_proto(cfg))
    print("wrote %s  (%d ToF sensors)" % (proto_path, 4 * cfg.tof.rays_per_zone))

    for layout in sorted(LAYOUTS):
        path = os.path.join(HERE, "worlds", "alvik_layout_%s.wbt" % layout)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(build_world(cfg, layout))
        print("wrote %s  (%d obstacles)" % (path, len(LAYOUTS[layout])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
