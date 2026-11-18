#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root"
tag=cluster-recovery-20261118
cache="${CANOPY_RECOVERY_CACHE:-$root/.recovery-cache}"
mkdir -p "$cache"
cache="$(cd "$cache" && pwd)"
fetch() {
  local file="$1"
  if [[ ! -f "$cache/$file" ]]; then
    curl --fail --location --retry 3 --output "$cache/$file.partial" \
      "https://github.com/fiberpine/Canopy/releases/download/$tag/$file"
    mv "$cache/$file.partial" "$cache/$file"
  fi
  (cd "$cache" && grep -F "  $file" "$root/recovery/assets.sha256" | sha256sum --strict --check)
}

case "${1:-}" in
  --kernel)
    . /etc/os-release
    [[ "$ID" == ubuntu && "$VERSION_ID" == 18.04 && "$(uname -m)" == x86_64 ]]
    fetch kernel-4.13.1-0521-learner.tar.gz
    if [[ "$(uname -r)" == 4.13.1-0521-learner ]]; then
      printf '%s\n' 'Kernel already running. Use --workspace to validate the checkout.'
      exit 0
    fi
    sudo tar --extract --gzip --file "$cache/kernel-4.13.1-0521-learner.tar.gz" --directory / \
      --exclude=lib/modules/4.13.1-0521-learner/build --exclude=lib/modules/4.13.1-0521-learner/source
    sudo depmod 4.13.1-0521-learner
    if [[ -f /boot/initrd.img-4.13.1-0521-learner ]]; then
      sudo update-initramfs -u -k 4.13.1-0521-learner
    else
      sudo update-initramfs -c -k 4.13.1-0521-learner
    fi
    sudo update-grub
    printf '%s\n' 'Use the VM console to boot 4.13.1-0521-learner and retain the vendor kernel as a boot option.'
    ;;
  --system)
    . /etc/os-release
    [[ "$ID" == ubuntu && "$VERSION_ID" == 18.04 && "$(uname -m)" == x86_64 ]]
    sudo apt-get update
    sudo apt-get install -y python3.6 python3.6-venv iproute2 iptables dnsmasq-base curl \
      initramfs-tools libstdc++6 libgomp1 libxcb-present0 libpango-1.0-0 \
      libpangocairo-1.0-0 libcairo2 libssl1.1
    for tool in mm-link mm-delay; do
      fetch "$tool"
      sudo install -o root -g root -m 4755 "$cache/$tool" "/usr/local/bin/$tool"
    done
    sudo install -m 0644 recovery/99-canopy-experiment.conf /etc/sysctl.d/99-canopy-experiment.conf
    sudo sysctl -p /etc/sysctl.d/99-canopy-experiment.conf
    ;;
  --workspace)
    fetch python36-tf114-venv.tar.gz
    fetch orca-server-joint-healthy
    if [[ ! -d runtime/venv ]]; then
      extract="$(mktemp -d "$root/runtime/venv-extract.XXXXXX")"
      tar -xzf "$cache/python36-tf114-venv.tar.gz" -C "$extract" --no-same-owner
      ln -sfn /usr/bin/python3.6 "$extract/venv/bin/python3"
      mv "$extract/venv" runtime/venv
      rmdir "$extract"
    fi
    CANOPY_BOOTSTRAP_REPORT="$cache/runtime-bootstrap.json" /usr/bin/python3.6 bootstrap_runtime.py
    native=frameworks/joint_healthy/native
    install -m 0755 "$cache/orca-server-joint-healthy" "$native/orca-server-joint-healthy"
    install -m 0644 recovery/native/orca-server-joint-healthy.build.json "$native/orca-server-joint-healthy.build.json"
    chmod +x frameworks/baseline/code/runtime/{client,orca-server-cloexec,orca-server-natural-cubic}
    runtime/venv/bin/python -B recovery/preflight.py
    ;;
  --verify)
    runtime/venv/bin/python -B recovery/preflight.py
    ;;
  *)
    echo "usage: bash recovery/restore.sh {--system|--kernel|--workspace|--verify}" >&2
    exit 2
    ;;
esac
