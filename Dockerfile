FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip iproute2 iputils-ping \
    iperf3 net-tools curl procps \
    && rm -rf /var/lib/apt/lists/*

COPY fl_simple_demo/ /fl/
WORKDIR /fl

RUN pip3 install --no-cache-dir \
    "flwr>=1.6.0" \
    "xgboost>=2.0.0" \
    "lightgbm>=4.0.0" \
    "catboost>=1.2.0" \
    "scikit-learn>=1.3.0" \
    "numpy>=1.24.0"

# Desabilita IPv6 e comportamento agressivo de ARP no boot
RUN echo "net.ipv6.conf.all.disable_ipv6=1" >> /etc/sysctl.conf && \
    echo "net.ipv6.conf.default.disable_ipv6=1" >> /etc/sysctl.conf && \
    echo "net.ipv4.conf.all.arp_announce=2" >> /etc/sysctl.conf && \
    echo "net.ipv4.conf.all.arp_ignore=1" >> /etc/sysctl.conf

# Script de entrada que aplica sysctl antes de qualquer coisa
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
