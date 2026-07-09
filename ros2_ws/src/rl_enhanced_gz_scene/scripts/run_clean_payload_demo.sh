#!/usr/bin/env bash

# 创建时间: 2026-05-29
# 创建者: Eugene
# 最后修改时间: 2026-06-04
# 功能: 在尽量干净的非 snap 污染环境中启动主包的载荷/无人机/箭头 Gazebo 演示。
# 主要输入: 可选 ros2 launch 参数，会透传给 rescue_playback_smooth.launch.py。
# 主要输出: 启动主包 Gazebo 播放链，避免 VS Code snap 环境变量干扰 GUI。
#
# Created: 2026-05-29
# Author: Eugene
# Last Modified: 2026-06-04
# Purpose: Launch the main payload/UAV/arrow Gazebo demo from a cleaned runtime
# environment so the GUI is less affected by snap-injected VS Code variables.
# Main Inputs: Optional ros2 launch arguments forwarded to rescue_playback_smooth.launch.py.
# Main Outputs: Starts the main Gazebo playback path with a cleaned GUI environment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${RL_ENHANCED_CLEAN_ENV_ACTIVE:-0}" != "1" ]]; then
  CLEAN_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  CLEAN_ENV=(
    RL_ENHANCED_CLEAN_ENV_ACTIVE=1
    HOME="${HOME:-/home/${USER:-eugene}}"
    USER="${USER:-${LOGNAME:-eugene}}"
    LOGNAME="${LOGNAME:-${USER:-eugene}}"
    SHELL="${SHELL:-/bin/bash}"
    PATH="$CLEAN_PATH"
    LANG="${LANG:-zh_CN.UTF-8}"
    TERM="${TERM:-xterm-256color}"
    DISPLAY="${DISPLAY:-}"
    XAUTHORITY="${XAUTHORITY:-}"
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}"
    XDG_SESSION_TYPE="${XDG_SESSION_TYPE:-x11}"
    DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-}"
  )
  if [[ "${RL_ENHANCED_USE_NVIDIA_PRIME:-0}" == "1" ]]; then
    CLEAN_ENV+=(
      __NV_PRIME_RENDER_OFFLOAD=1
      __GLX_VENDOR_LIBRARY_NAME=nvidia
      __VK_LAYER_NV_optimus=NVIDIA_only
    )
  fi
  if [[ "${RL_ENHANCED_FORCE_NVIDIA_EGL:-0}" == "1" ]]; then
    CLEAN_ENV+=(
      __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
      __GLX_VENDOR_LIBRARY_NAME=nvidia
    )
  fi
  exec /usr/bin/env -i "${CLEAN_ENV[@]}" "$0" "$@"
fi

find_workspace_root() {
  local start_dir="$1"
  local current="$start_dir"
  while [[ "$current" != "/" ]]; do
    if [[ -f "$current/install/local_setup.bash" ]]; then
      printf '%s\n' "$current"
      return 0
    fi
    current="$(dirname -- "$current")"
  done
  return 1
}

restore_or_unset() {
  local name="$1"
  local backup_name="${name}_VSCODE_SNAP_ORIG"
  if [[ -v "$backup_name" ]]; then
    local backup_value="${!backup_name}"
    if [[ -n "$backup_value" ]]; then
      export "$name=$backup_value"
    else
      unset "$name"
    fi
  else
    unset "$name"
  fi
}

if ! command -v ros2 >/dev/null 2>&1; then
  if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    # Source ROS 2 first so ros2 / gz tools are available even from a clean shell.
    # shellcheck disable=SC1091
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
  fi
fi

WORKSPACE_ROOT="$(find_workspace_root "$SCRIPT_DIR")"
if [[ -z "$WORKSPACE_ROOT" ]]; then
  echo "[run_clean_payload_demo] failed to locate ros2_ws workspace root" >&2
  exit 1
fi

if [[ -f "$WORKSPACE_ROOT/install/local_setup.bash" ]]; then
  # Load the local workspace so the main package version is used instead of stale installs.
  # shellcheck disable=SC1091
  set +u
  source "$WORKSPACE_ROOT/install/local_setup.bash"
  set -u
fi

DEFAULT_WORLD_PATH="$WORKSPACE_ROOT/src/rl_enhanced_gz_scene/worlds/payload_uav_only_demo_world.sdf"
if [[ -f "$WORKSPACE_ROOT/install/rl_enhanced_gz_scene/share/rl_enhanced_gz_scene/worlds/payload_uav_only_demo_world.sdf" ]]; then
  DEFAULT_WORLD_PATH="$WORKSPACE_ROOT/install/rl_enhanced_gz_scene/share/rl_enhanced_gz_scene/worlds/payload_uav_only_demo_world.sdf"
fi

restore_or_unset GDK_BACKEND
restore_or_unset GIO_MODULE_DIR
restore_or_unset GSETTINGS_SCHEMA_DIR
restore_or_unset GTK_EXE_PREFIX
restore_or_unset GTK_IM_MODULE_FILE
restore_or_unset GTK_PATH
restore_or_unset LOCPATH
restore_or_unset XDG_DATA_HOME
restore_or_unset XDG_DATA_DIRS
restore_or_unset XDG_CONFIG_DIRS

unset QT_QPA_PLATFORM

DEFAULT_ARGS=(
  auto_run:=false
  gui:=true
  use_inference_npz:=true
  spawn_payload_entity:=false
  world_path:=$DEFAULT_WORLD_PATH
)

echo "[run_clean_payload_demo] workspace: $WORKSPACE_ROOT"
echo "[run_clean_payload_demo] launching main playback path with cleaned GUI env"

exec ros2 launch rl_enhanced_gz_scene rescue_playback_smooth.launch.py "${DEFAULT_ARGS[@]}" "$@"