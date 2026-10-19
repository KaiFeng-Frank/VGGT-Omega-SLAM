#!/usr/bin/env bash
# Optional cgroup RAM limit for desktop users; plain env is portable on Linux.
# Set VOS_SYSTEMD_MEMORY_MAX=12G to opt into a systemd user scope.
run_env() {
  if [[ -n "${VOS_SYSTEMD_MEMORY_MAX:-}" ]]; then
    systemd-run --user --scope --quiet -p "MemoryMax=$VOS_SYSTEMD_MEMORY_MAX" env "$@"
  else
    env "$@"
  fi
}
