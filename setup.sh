#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ "${1:-}" == "--system" ]]; then
  sudo apt-get update
  sudo apt-get install -y python3.6 python3.6-venv build-essential autoconf automake libtool pkg-config iproute2 iptables protobuf-compiler libprotobuf-dev libssl-dev dnsmasq-base apache2 apache2-dev libcairo2-dev libpango1.0-dev libxcb-present-dev
  (cd runtime/mahimahi && autoreconf -i && ./configure && make -j2 && sudo make install)
  sudo dpkg -i runtime/kernel/*.deb
  printf '%s\n' 'Custom kernel installed. Reboot into 4.13.1-0521-learner before TCP training; this script does not reboot.'
fi
/usr/bin/python3.6 bootstrap_runtime.py
runtime/venv/bin/python run.py doctor
