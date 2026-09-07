# Copy to the board as wifi_secrets.py and fill in. Never commit the real file.
#
# PC_IP must be the address of the laptop running run_diff_qp_real.py: the
# board drops every packet from any other source (read_packet in main.py), so a
# stale value here looks exactly like a dead controller - telemetry arrives, no
# command is ever accepted, and the robot brakes on the 300 ms timeout.
# Check it with `ipconfig` on the same Wi-Fi network before every session; a
# DHCP lease renewal is enough to break it.

WIFI_SSID = "your-2.4GHz-ssid"
WIFI_PASSWORD = "your-password"
PC_IP = "192.168.1.100"
