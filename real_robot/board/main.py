from arduino_alvik import ArduinoAlvik
import network
import socket
from time import sleep_ms, ticks_ms, ticks_diff

from wifi_secrets import WIFI_SSID, WIFI_PASSWORD, PC_IP

# Keep this file ASCII-only for reliable Arduino Lab Raw REPL uploads.

# Network settings
COMMAND_PORT = 4212
PC_TELEMETRY_PORT = 4211
WIFI_TIMEOUT_MS = 20000
TELEMETRY_INTERVAL_MS = 100

# Motion and safety limits
# The PC sends calibrated motor commands. 3.4 cm/s and 30 deg/s are command
# caps, not the desired body speeds. They allow the PC to compensate measured
# gains so that the body remains close to 3 cm/s and 25 deg/s.
MAX_LINEAR_CM_S = 3.4
MAX_ANGULAR_DEG_S = 30.0
COMMAND_TIMEOUT_MS = 300
HARD_STOP_FRONT_MM = 100.0
HARD_STOP_RELEASE_MM = 110.0
CRITICAL_STOP_FRONT_MM = 60.0
TOF_TO_BUMPER_MM = 8.0
TOF_SAFETY_SAMPLE_MS = 100
TOF_CONSECUTIVE_FRAMES = 3

alvik = ArduinoAlvik()


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def update_tof_safety(
    CL,
    C,
    CR,
    low_counts,
    release_count,
    stop_active,
    critical_active
):
    front_values = (CL, C, CR)

    # Count consecutive low samples independently for CL, C and CR.
    for index in range(3):
        if front_values[index] <= HARD_STOP_FRONT_MM:
            if low_counts[index] < TOF_CONSECUTIVE_FRAMES:
                low_counts[index] += 1
        else:
            low_counts[index] = 0

    # A very close return remains an immediate one-sample stop.
    if min(front_values) <= CRITICAL_STOP_FRONT_MM:
        return True, True, "HARD_TOF_STOP_CRITICAL", 0

    if stop_active:
        # Release only after every front channel is safely above the release
        # threshold for three consecutive samples. This prevents chatter.
        if min(front_values) >= HARD_STOP_RELEASE_MM:
            release_count += 1
        else:
            release_count = 0

        if release_count >= TOF_CONSECUTIVE_FRAMES:
            low_counts[0] = 0
            low_counts[1] = 0
            low_counts[2] = 0
            return False, False, "TOF_STOP_RELEASED", 0

        # Keep a critical stop fully latched until the normal three-frame
        # release condition is satisfied. A noisy 60-to-61 mm transition must
        # not re-enable rotation while the obstacle is still extremely close.
        if critical_active:
            return True, True, "HARD_TOF_STOP_CRITICAL", release_count

        return True, False, "HARD_TOF_STOP_CONFIRMED", release_count

    if max(low_counts) >= TOF_CONSECUTIVE_FRAMES:
        return True, False, "HARD_TOF_STOP_CONFIRMED", 0

    if max(low_counts) > 0:
        return False, False, "TOF_TRANSIENT_LOW_IGNORED", 0

    return False, False, "TOF_CLEAR", 0


def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if wlan.isconnected():
        return wlan

    try:
        wlan.disconnect()
    except OSError:
        pass

    print("WIFI_CONNECTING")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    start_ms = ticks_ms()

    while not wlan.isconnected():
        if ticks_diff(ticks_ms(), start_ms) > WIFI_TIMEOUT_MS:
            raise RuntimeError("WIFI_CONNECTION_FAILED")
        sleep_ms(250)

    print("WIFI_CONNECTED")
    print("ALVIK_IP=" + wlan.ifconfig()[0])
    return wlan


def read_packet(udp):
    try:
        data, address = udp.recvfrom(160)
    except OSError:
        return None

    if address[0] != PC_IP:
        return None

    try:
        fields = data.decode().strip().split(",")
        packet_type = fields[0]

        if packet_type == "CMD" and len(fields) == 4:
            sequence = int(fields[1])
            linear_cm_s = clamp(
                float(fields[2]),
                0.0,
                MAX_LINEAR_CM_S
            )
            angular_deg_s = clamp(
                float(fields[3]),
                -MAX_ANGULAR_DEG_S,
                MAX_ANGULAR_DEG_S
            )
            return (
                "CMD",
                sequence,
                linear_cm_s,
                angular_deg_s
            )

        if packet_type == "RESET" and len(fields) == 5:
            sequence = int(fields[1])
            x_mm = float(fields[2])
            y_mm = float(fields[3])
            theta_deg = float(fields[4])
            return (
                "RESET",
                sequence,
                x_mm,
                y_mm,
                theta_deg
            )

        if packet_type == "BRAKE" and len(fields) == 2:
            sequence = int(fields[1])
            return ("BRAKE", sequence)

    except (ValueError, IndexError):
        return None

    return None


def send_telemetry(
    udp,
    telemetry_sequence,
    x_mm,
    y_mm,
    theta_deg,
    L,
    CL,
    C,
    CR,
    R,
    command_v,
    command_w,
    feedback_v,
    feedback_w,
    safety_state,
    last_command_sequence
):
    C_front = C - TOF_TO_BUMPER_MM

    # Packet fields:
    # TEL,telemetry_seq,board_ms,x,y,theta,L,CL,C,CR,R,C_front,
    # command_v,command_w,feedback_v,feedback_w,state,last_command_seq
    packet = (
        "TEL," +
        str(telemetry_sequence) + "," +
        str(ticks_ms()) + "," +
        str(x_mm) + "," +
        str(y_mm) + "," +
        str(theta_deg) + "," +
        str(L) + "," +
        str(CL) + "," +
        str(C) + "," +
        str(CR) + "," +
        str(R) + "," +
        str(C_front) + "," +
        str(command_v) + "," +
        str(command_w) + "," +
        str(feedback_v) + "," +
        str(feedback_w) + "," +
        safety_state + "," +
        str(last_command_sequence)
    )

    try:
        udp.sendto(packet.encode(), (PC_IP, PC_TELEMETRY_PORT))
    except OSError:
        pass


try:
    wlan = connect_wifi()

    alvik.begin()
    alvik.brake()
    sleep_ms(1500)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("0.0.0.0", COMMAND_PORT))
    udp.settimeout(0.01)

    command_v = 0.0
    command_w = 0.0
    applied_v = 0.0
    applied_w = 0.0
    last_command_ms = None
    last_command_sequence = -1
    telemetry_sequence = 0
    last_telemetry_ms = ticks_ms()
    last_tof_safety_ms = None
    tof_low_counts = [0, 0, 0]
    tof_release_count = 0
    tof_stop_active = False
    tof_critical_active = False
    tof_filter_state = "TOF_CLEAR"
    safety_state = "STARTUP_STOP"

    print("POSE_TOF_BRIDGE_READY")
    print("COMMAND_PORT=" + str(COMMAND_PORT))

    while True:
        if not wlan.isconnected():
            alvik.brake()
            applied_v = 0.0
            applied_w = 0.0
            safety_state = "WIFI_RECONNECT_STOP"
            wlan = connect_wifi()

        packet = read_packet(udp)

        if packet is not None:
            packet_type = packet[0]
            packet_sequence = packet[1]

            if packet_sequence > last_command_sequence:
                last_command_sequence = packet_sequence

                if packet_type == "CMD":
                    command_v = packet[2]
                    command_w = packet[3]
                    last_command_ms = ticks_ms()

                elif packet_type == "RESET":
                    alvik.brake()
                    command_v = 0.0
                    command_w = 0.0
                    applied_v = 0.0
                    applied_w = 0.0
                    alvik.reset_pose(
                        packet[2],
                        packet[3],
                        packet[4],
                        "mm",
                        "deg"
                    )
                    last_command_ms = ticks_ms()
                    safety_state = "POSE_RESET"

                elif packet_type == "BRAKE":
                    alvik.brake()
                    command_v = 0.0
                    command_w = 0.0
                    applied_v = 0.0
                    applied_w = 0.0
                    last_command_ms = ticks_ms()
                    safety_state = "REMOTE_BRAKE"

        L, CL, C, CR, R = alvik.get_distance("mm")
        x_mm, y_mm, theta_deg = alvik.get_pose("mm", "deg")
        now_ms = ticks_ms()

        if None in (L, CL, C, CR, R, x_mm, y_mm, theta_deg):
            alvik.brake()
            applied_v = 0.0
            applied_w = 0.0
            safety_state = "SENSOR_NOT_READY"

        else:
            # Evaluate only once per exported ToF control frame. Counting the
            # same cached sensor value in the 10 ms loop would defeat the
            # multi-frame filter.
            if (last_tof_safety_ms is None or
                    ticks_diff(now_ms, last_tof_safety_ms) >=
                    TOF_SAFETY_SAMPLE_MS):
                (tof_stop_active,
                 tof_critical_active,
                 tof_filter_state,
                 tof_release_count) = update_tof_safety(
                    CL,
                    C,
                    CR,
                    tof_low_counts,
                    tof_release_count,
                    tof_stop_active,
                    tof_critical_active
                 )
                last_tof_safety_ms = now_ms

            command_is_stale = (
                last_command_ms is None or
                ticks_diff(now_ms, last_command_ms) > COMMAND_TIMEOUT_MS
            )

            if command_is_stale:
                alvik.brake()
                applied_v = 0.0
                applied_w = 0.0
                safety_state = "COMMAND_TIMEOUT_STOP"

            elif tof_critical_active:
                alvik.brake()
                applied_v = 0.0
                applied_w = 0.0
                safety_state = tof_filter_state

            elif tof_stop_active:
                # A confirmed 100 mm obstacle blocks forward motion, but an
                # in-place turn is still allowed. With the circular footprint,
                # rotation does not enlarge the occupied disk and lets the PC
                # controller steer the front ToF away from the obstacle.
                alvik.drive(0.0, command_w, "cm/s", "deg/s")
                applied_v = 0.0
                applied_w = command_w
                safety_state = "TOF_FORWARD_BLOCKED"

            else:
                alvik.drive(command_v, command_w, "cm/s", "deg/s")
                applied_v = command_v
                applied_w = command_w
                if tof_filter_state == "TOF_TRANSIENT_LOW_IGNORED":
                    safety_state = tof_filter_state
                else:
                    safety_state = "COMMAND_ACTIVE"

            if ticks_diff(now_ms, last_telemetry_ms) >= TELEMETRY_INTERVAL_MS:
                try:
                    feedback_v, feedback_w = alvik.get_drive_speed(
                        "cm/s",
                        "deg/s"
                    )
                except Exception:
                    # Compatibility fallback for an older board library.
                    feedback_v = applied_v
                    feedback_w = applied_w
                if feedback_v is None:
                    feedback_v = applied_v
                if feedback_w is None:
                    feedback_w = applied_w
                telemetry_sequence += 1
                send_telemetry(
                    udp,
                    telemetry_sequence,
                    x_mm,
                    y_mm,
                    theta_deg,
                    L,
                    CL,
                    C,
                    CR,
                    R,
                    command_v,
                    command_w,
                    feedback_v,
                    feedback_w,
                    safety_state,
                    last_command_sequence
                )
                last_telemetry_ms = now_ms

        sleep_ms(10)

except KeyboardInterrupt:
    print("BRIDGE_STOPPED")

except Exception as e:
    print("ERROR=" + str(e))

finally:
    alvik.brake()
