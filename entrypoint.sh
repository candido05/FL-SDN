#!/bin/bash
sysctl -p /etc/sysctl.conf 2>/dev/null
ip link set eth0 down
exec /bin/bash
