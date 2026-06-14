#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "$(basename "$SCRIPT_DIR")" == "launcher" && -d "$SCRIPT_DIR/../profiles" ]]; then
  MANAGER_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
else
  MANAGER_ROOT="$SCRIPT_DIR"
fi
RUNTIME_ROOT=${RUNTIME_ROOT:-"$MANAGER_ROOT"}
PROFILE_DIR=${PROFILE_DIR:-"$MANAGER_ROOT/profiles"}
TEMPLATE_DIR=${TEMPLATE_DIR:-"$PROFILE_DIR/templates"}
LOG_DIR=${LOG_DIR:-"$MANAGER_ROOT/run-logs"}
STATE_FILE=${STATE_FILE:-"$LOG_DIR/start-manager.state"}
STAMP=$(date +%Y%m%d-%H%M%S)
VERSION=${VERSION:-0.1.7}

banner() {
  cat <<EOF
============================================================
 vLLM 2080 Ti Definitive Edition v$VERSION
 Service manager
============================================================
EOF
}

pid_is_running() {
  local pid=${1:-}
  [[ -n "$pid" && -d "/proc/$pid" ]]
}

pid_file_service_name() {
  local pid_file=$1
  local name
  name=$(basename "$pid_file" .pid)
  name=${name#vllm-}
  printf '%s\n' "$name"
}

current_service_info() {
  local pid_file pid name

  if [[ -n "${LAST_PID_FILE:-}" && -f "${LAST_PID_FILE:-}" ]]; then
    pid=$(cat "$LAST_PID_FILE" 2>/dev/null || true)
    if pid_is_running "$pid"; then
      name=$(pid_file_service_name "$LAST_PID_FILE")
      printf '%s\t%s\t%s\n' "$LAST_PID_FILE" "$pid" "$name"
      return 0
    fi
  fi

  [[ -d "$LOG_DIR" ]] || return 1
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if pid_is_running "$pid"; then
      name=$(pid_file_service_name "$pid_file")
      printf '%s\t%s\t%s\n' "$pid_file" "$pid" "$name"
      return 0
    fi
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)
  return 1
}

render_service_status() {
  local info pid_file pid name api

  echo "Service status"
  if info=$(current_service_info); then
    IFS=$'\t' read -r pid_file pid name <<< "$info"
    api=${LAST_API_LAN:-${LAST_API_LOCAL:-http://127.0.0.1:${PORT:-8000}/v1}}
    printf '  Status:  RUNNING\n'
    printf '  Model:   %s\n' "${SERVED_NAME:-$name}"
    printf '  API:     %s\n' "$api"
    printf '  PID:     %s\n' "$pid"
  else
    printf '  Status:  STOPPED\n'
  fi
  echo
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_tty() {
  [[ -t 0 && -t 1 ]] || { : </dev/tty >/dev/tty; } 2>/dev/null
}

pause_enter() {
  is_tty || return 0
  read -r -p "Press Enter to continue..." _
}

normalize_bool() {
  case "${1,,}" in
    1|yes|y|true|on) echo 1 ;;
    *) echo 0 ;;
  esac
}

read_profile_value() {
  local file=$1
  local key=$2
  awk -F= -v key="$key" '
    $1 == key {
      value = substr($0, index($0, "=") + 1)
      gsub(/^[ \t]+|[ \t]+$/, "", value)
      gsub(/^'\''|'\''$/, "", value)
      gsub(/^"|"$/, "", value)
      print value
      exit
    }
  ' "$file"
}

profile_key_is_global() {
  case "$1" in
    MODEL_DIR|PROFILE_DIR|PROFILE|MODE|PORT|SERVICE_SCOPE|GPU_DEVICES|TP_SIZE|\
CHAT_TEMPLATE_FILE|CHAT_TEMPLATE_PRESET|TEMPLATE_DIR|REASONING_PARSER|\
DEFAULT_CHAT_TEMPLATE_KWARGS|REASONING_MODE|REASONING_BUDGET|\
VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

source_profile_defaults() {
  local file=$1
  [[ -f "$file" ]] || return 0

  local key value
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    profile_key_is_global "$key" && continue
    if [[ -v "$key" ]]; then
      continue
    fi
    value=$(read_profile_value "$file" "$key")
    printf -v "$key" '%s' "$value"
    export "$key"
  done < <(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$file" | sort -u)
}

apply_profile_overrides() {
  local file=$1
  [[ -f "$file" ]] || return 0

  local key value
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    profile_key_is_global "$key" && continue
    value=$(read_profile_value "$file" "$key")
    printf -v "$key" '%s' "$value"
    export "$key"
  done < <(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$file" | sort -u)
}

resolve_profile_file() {
  if [[ -n "${PROFILE_FILE:-}" ]]; then
    printf '%s\n' "$PROFILE_FILE"
    return 0
  fi
  if [[ -z "${PROFILE:-}" ]]; then
    return 1
  fi
  if [[ -f "$PROFILE_DIR/$PROFILE" ]]; then
    printf '%s\n' "$PROFILE_DIR/$PROFILE"
    return 0
  fi
  if [[ -f "$PROFILE_DIR/${PROFILE%.env}.env" ]]; then
    printf '%s\n' "$PROFILE_DIR/${PROFILE%.env}.env"
    return 0
  fi
  return 1
}

load_manager_state() {
  [[ -f "$STATE_FILE" ]] || return 0
  # shellcheck disable=SC1090
  source "$STATE_FILE"
}

save_manager_state() {
  mkdir -p "$LOG_DIR"
  {
    printf 'MODEL_DIR=%q\n' "${MODEL_DIR:-}"
    printf 'MODEL_SEARCH_PATHS=%q\n' "${MODEL_SEARCH_PATHS:-/home/david/work/models}"
    printf 'PROFILE_DIR=%q\n' "${PROFILE_DIR:-}"
    printf 'TEMPLATE_DIR=%q\n' "${TEMPLATE_DIR:-}"
    printf 'PROFILE=%q\n' "${PROFILE:-}"
    printf 'MODEL_FAMILY=%q\n' "${MODEL_FAMILY:-}"
    printf 'PROFILE_GROUP=%q\n' "${PROFILE_GROUP:-}"
    printf 'MODEL_VARIANT=%q\n' "${MODEL_VARIANT:-}"
    printf 'SERVED_NAME=%q\n' "${SERVED_NAME:-}"
    printf 'GPU_DEVICES=%q\n' "${GPU_DEVICES:-}"
    printf 'QUANTIZATION=%q\n' "${QUANTIZATION:-}"
    printf 'KV_CACHE_DTYPE=%q\n' "${KV_CACHE_DTYPE:-}"
    printf 'MAX_MODEL_LEN=%q\n' "${MAX_MODEL_LEN:-}"
    printf 'GPU_UTIL=%q\n' "${GPU_UTIL:-}"
    printf 'MAX_BATCHED_TOKENS=%q\n' "${MAX_BATCHED_TOKENS:-}"
    printf 'MAX_NUM_SEQS=%q\n' "${MAX_NUM_SEQS:-}"
    printf 'MTP_K=%q\n' "${MTP_K:-}"
    printf 'MESSAGE_TYPE=%q\n' "${MESSAGE_TYPE:-}"
    printf 'MM_LIMIT_JSON=%q\n' "${MM_LIMIT_JSON:-}"
    printf 'LANGUAGE_MODEL_ONLY=%q\n' "${LANGUAGE_MODEL_ONLY:-}"
    printf 'SKIP_MM_PROFILING=%q\n' "${SKIP_MM_PROFILING:-}"
    printf 'HF_OVERRIDES_JSON=%q\n' "${HF_OVERRIDES_JSON:-}"
    printf 'ADDITIONAL_CONFIG_JSON=%q\n' "${ADDITIONAL_CONFIG_JSON:-}"
    printf 'SPECULATIVE_CONFIG=%q\n' "${SPECULATIVE_CONFIG:-}"
    printf 'COMPILATION_CONFIG_JSON=%q\n' "${COMPILATION_CONFIG_JSON:-}"
    printf 'TP_SIZE=%q\n' "${TP_SIZE:-}"
    printf 'CHAT_TEMPLATE_FILE=%q\n' "${CHAT_TEMPLATE_FILE:-}"
    printf 'CHAT_TEMPLATE_PRESET=%q\n' "${CHAT_TEMPLATE_PRESET:-}"
    printf 'ATTENTION_BACKEND=%q\n' "${ATTENTION_BACKEND:-}"
    printf 'REASONING_MODE=%q\n' "${REASONING_MODE:-}"
    printf 'REASONING_PARSER=%q\n' "${REASONING_PARSER:-}"
    printf 'REASONING_BUDGET=%q\n' "${REASONING_BUDGET:-}"
    printf 'DEFAULT_CHAT_TEMPLATE_KWARGS=%q\n' "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
    printf 'ENFORCE_EAGER=%q\n' "${ENFORCE_EAGER:-}"
    printf 'NO_ASYNC_SCHEDULING=%q\n' "${NO_ASYNC_SCHEDULING:-}"
    printf 'DISABLE_HYBRID_KV_CACHE_MANAGER=%q\n' "${DISABLE_HYBRID_KV_CACHE_MANAGER:-}"
    printf 'DISABLE_PREFIX_CACHING=%q\n' "${DISABLE_PREFIX_CACHING:-}"
    printf 'DISABLE_CUSTOM_ALL_REDUCE=%q\n' "${DISABLE_CUSTOM_ALL_REDUCE:-}"
    printf 'DISABLE_LOG_STATS=%q\n' "${DISABLE_LOG_STATS:-}"
    printf 'ENABLE_TOOL_CALLING=%q\n' "${ENABLE_TOOL_CALLING:-}"
    printf 'TOOL_CALL_PARSER=%q\n' "${TOOL_CALL_PARSER:-}"
    printf 'VLLM_ENGINE_READY_TIMEOUT_S=%q\n' "${VLLM_ENGINE_READY_TIMEOUT_S:-}"
    printf 'OMP_NUM_THREADS=%q\n' "${OMP_NUM_THREADS:-}"
    printf 'VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=%q\n' "${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-}"
    printf 'MODE=%q\n' "${MODE:-safe}"
    printf 'PORT=%q\n' "${PORT:-8000}"
    printf 'SERVICE_SCOPE=%q\n' "${SERVICE_SCOPE:-local}"
    printf 'LAST_PID_FILE=%q\n' "${LAST_PID_FILE:-}"
    printf 'LAST_LOG_FILE=%q\n' "${LAST_LOG_FILE:-}"
    printf 'LAST_API_LOCAL=%q\n' "${LAST_API_LOCAL:-}"
    printf 'LAST_API_LAN=%q\n' "${LAST_API_LAN:-}"
    printf 'LAST_SMOKE_OUTPUT=%q\n' "${LAST_SMOKE_OUTPUT:-}"
  } > "$STATE_FILE"
}

list_profile_groups() {
  [[ -d "$PROFILE_DIR" ]] || return 0
  find "$PROFILE_DIR" -mindepth 1 -maxdepth 1 -type d ! -name templates ! -name user -printf '%f\n' | sort
}

list_profiles_in_group() {
  local group=$1
  [[ -d "$PROFILE_DIR/$group" ]] || return 0
  find "$PROFILE_DIR/$group" -type f -name '*.env' -printf '%P\n' |
    awk -v include_experimental="${PROFILE_INCLUDE_EXPERIMENTAL:-0}" '
      include_experimental == "1" || $0 !~ /(^|\/)experimental\//
    ' |
    sort
}

list_profiles() {
  [[ -d "$PROFILE_DIR" ]] || return 0
  find "$PROFILE_DIR" -type f -name '*.env' -printf '%P\n' |
    awk -v include_experimental="${PROFILE_INCLUDE_EXPERIMENTAL:-0}" '
      include_experimental == "1" || $0 !~ /(^|\/)experimental\//
    ' |
    sort
}

list_profiles_for_model() {
  local family=$1
  local quantization=${2:-}
  local profile profile_file profile_family profile_variant

  while IFS= read -r profile; do
    [[ -n "$profile" ]] || continue
    profile_file="$PROFILE_DIR/$profile"
    profile_family=$(read_profile_value "$profile_file" MODEL_FAMILY)
    profile_variant=$(read_profile_value "$profile_file" MODEL_VARIANT)

    if [[ "$family" == qwen* && -n "$profile_family" && "$profile_family" != qwen* ]]; then
      continue
    fi
    if [[ "$family" == gemma* && -n "$profile_family" && "$profile_family" != gemma* ]]; then
      continue
    fi

    if [[ "$family" == qwen* && "$quantization" == fp8 && "$profile_variant" != fp8 ]]; then
      continue
    fi
    if [[ "$family" == qwen* && -n "$quantization" && "$quantization" != fp8 && "$profile_variant" == fp8 ]]; then
      continue
    fi

    printf '%s\n' "$profile"
  done < <(list_profiles)
}

first_compatible_mode() {
  local compatible_modes=${1:-safe,normal,fast}
  local candidate
  for candidate in ${compatible_modes//,/ }; do
    candidate=${candidate//[[:space:]]/}
    case "$candidate" in
      stable) echo safe; return 0 ;;
      speed) echo normal; return 0 ;;
      aggressive) echo fast; return 0 ;;
      safe|normal|fast) echo "$candidate"; return 0 ;;
    esac
  done
  echo safe
}

mode_is_compatible() {
  local mode=$1
  local compatible_modes=${2:-safe,normal,fast}
  local candidate
  for candidate in ${compatible_modes//,/ }; do
    candidate=${candidate//[[:space:]]/}
    case "$candidate" in
      stable) candidate=safe ;;
      speed) candidate=normal ;;
      aggressive) candidate=fast ;;
    esac
    [[ "$candidate" == "$mode" ]] && return 0
  done
  return 1
}

profile_family_dir() {
  if [[ -n "${PROFILE:-}" && "$PROFILE" == */* ]]; then
    printf '%s\n' "${PROFILE%%/*}"
    return 0
  fi
  case "${MODEL_FAMILY:-}" in
    gemma*) echo gemma31b ;;
    qwen*|"") echo qwopus36-27b ;;
    *)
      printf '%s\n' "${MODEL_FAMILY//[^A-Za-z0-9_.-]/-}"
      ;;
  esac
}

profile_compatible_modes_for_current() {
  normalize_mode
  local mode=${MODE:-safe}
  local kv=${KV_CACHE_DTYPE:-}
  local mtp=${MTP_K:-0}

  if [[ "$mode" == "safe" ]]; then
    case "$kv" in
      ""|fp16|default|auto)
        ;;
      *)
        if [[ "$mtp" =~ ^[0-9]+$ ]] && (( mtp > 0 )); then
          echo fast
          return 0
        fi
        ;;
    esac
  fi
  echo "$mode"
}

sanitize_profile_name() {
  local name=$1
  name=${name%.env}
  name=$(printf '%s' "$name" | sed -E 's/[^A-Za-z0-9_.-]+/-/g; s/^-+//; s/-+$//')
  [[ -n "$name" ]] || name="user-profile"
  printf '%s\n' "$name"
}

write_profile_entry() {
  local file=$1
  local key=$2
  local value=${3:-}
  [[ -n "$value" ]] || return 0
  value=${value//\'/}
  printf "%s='%s'\n" "$key" "$value" >> "$file"
}

list_template_presets() {
  [[ -d "$TEMPLATE_DIR" ]] || return 0
  find "$TEMPLATE_DIR" -type f \( -name '*.jinja' -o -name '*.jinja2' -o -name '*.txt' \) -printf '%P\n' |
    sort
}

resolve_template_file() {
  local template=${1:-}
  [[ -n "$template" ]] || return 1
  if [[ -f "$template" ]]; then
    printf '%s\n' "$template"
    return 0
  fi
  if [[ -f "$TEMPLATE_DIR/$template" ]]; then
    printf '%s\n' "$TEMPLATE_DIR/$template"
    return 0
  fi
  return 1
}

current_template_label() {
  if [[ -n "${CHAT_TEMPLATE_PRESET:-}" ]]; then
    printf '%s\n' "$CHAT_TEMPLATE_PRESET"
  elif [[ -n "${CHAT_TEMPLATE_FILE:-}" ]]; then
    printf '%s\n' "$CHAT_TEMPLATE_FILE"
  else
    printf 'model default'
  fi
}

current_reasoning_label() {
  local label="template default"
  if [[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]]; then
    case "${DEFAULT_CHAT_TEMPLATE_KWARGS//[[:space:]]/}" in
      *'"enable_thinking":false'*|*"\"enable_thinking\":false"*)
        label="thinking off"
        ;;
      *'"enable_thinking":true'*|*"\"enable_thinking\":true"*)
        label="thinking on"
        ;;
      *)
        label="custom kwargs"
        ;;
    esac
  fi
  if [[ -n "${REASONING_PARSER:-}" ]]; then
    label+=" / parser=$REASONING_PARSER"
  fi
  if [[ -n "${REASONING_BUDGET:-}" ]]; then
    label+=" / default budget=$REASONING_BUDGET"
  fi
  printf '%s\n' "$label"
}

gpu_device_count() {
  local devices=${1:-}
  local count=0 part
  devices=${devices// /}
  [[ -n "$devices" ]] || {
    echo 0
    return 0
  }
  IFS=',' read -r -a parts <<< "$devices"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] && count=$((count + 1))
  done
  echo "$count"
}

list_nvidia_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null |
    awk -F, '
      {
        idx = $1
        name = substr($0, index($0, ",") + 1)
        gsub(/^[ \t]+|[ \t]+$/, "", idx)
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        if (idx != "" && name != "") {
          print idx "\t" name
        }
      }
    '
}

profile_summary() {
  local profile_file=$1
  [[ -f "$profile_file" ]] || return 0

  local keys=(
    SERVED_NAME
    COMPATIBLE_MODES
    MODEL_FAMILY
    PROFILE_GROUP
    MODEL_VARIANT
    QUANTIZATION
    KV_CACHE_DTYPE
    MAX_MODEL_LEN
    GPU_UTIL
    MAX_BATCHED_TOKENS
    MAX_NUM_SEQS
    MTP_K
    MM_LIMIT_JSON
    HF_OVERRIDES_JSON
  )

  local key value
  for key in "${keys[@]}"; do
    value=$(read_profile_value "$profile_file" "$key")
    [[ -n "$value" ]] || value="-"
    printf '  %-22s %s\n' "$key" "$value"
  done
}

menu_select() {
  local title=$1
  local default=$2
  shift
  shift
  local options=("$@")
  local count=${#options[@]}
  local idx=0
  local key answer answer_rest selected_index number_buffer=""

  (( count > 0 )) || return 1
  for i in "${!options[@]}"; do
    if [[ "${options[$i]}" == "$default" ]]; then
      idx=$i
      break
    fi
  done

  if ! is_tty; then
    printf '%s\n' "${options[$idx]}"
    return 0
  fi

  while true; do
    clear >/dev/tty
    {
      banner
      echo "$title"
      echo
      for i in "${!options[@]}"; do
        if (( i == idx )); then
          printf ' > %d. %s\n' "$((i + 1))" "${options[$i]}"
        else
          printf '   %d. %s\n' "$((i + 1))" "${options[$i]}"
        fi
      done
      echo
      if (( count >= 10 )); then
        echo "Use Up/Down, Enter to select. Type a number then Enter for 10+ items. Esc returns."
        [[ -n "$number_buffer" ]] && echo "Input: $number_buffer"
      else
        echo "Press a number to select, Enter for the highlighted item."
      fi
    } >/dev/tty

    if (( count >= 10 )); then
      printf 'Select [1-%s]: ' "$count" >/dev/tty
      IFS= read -rsn1 key </dev/tty || true
      printf '\n' >/dev/tty
      [[ "$key" == $'\x04' ]] && return 1

      if [[ -z "$key" ]]; then
        if [[ -n "$number_buffer" ]]; then
          if [[ "$number_buffer" =~ ^[0-9]+$ ]] && (( number_buffer >= 1 && number_buffer <= count )); then
            printf '%s\n' "${options[$((number_buffer - 1))]}"
            return 0
          fi
          echo "Please enter a listed number." >&2
          number_buffer=""
          sleep 1
          continue
        fi
        printf '%s\n' "${options[$idx]}"
        return 0
      fi

      if [[ "$key" == $'\x1b' ]]; then
        read -rsn2 -t 0.1 key </dev/tty || true
        if [[ -z "$key" ]]; then
          return 1
        fi
        case "$key" in
          "[A") (( idx > 0 )) && idx=$((idx - 1)) ;;
          "[B") (( idx < count - 1 )) && idx=$((idx + 1)) ;;
        esac
        continue
      fi

      if [[ "$key" =~ ^[0-9]$ ]]; then
        number_buffer+="$key"
        if (( number_buffer > count )); then
          echo "Please enter a listed number." >&2
          number_buffer=""
          sleep 1
        fi
        continue
      fi

      if [[ "$key" == $'\x7f' || "$key" == $'\b' ]]; then
        number_buffer=${number_buffer%?}
        continue
      fi

      answer="$number_buffer$key"
      number_buffer=""
      read -r -t 0.5 answer_rest </dev/tty || answer_rest=""
      answer="${answer}${answer_rest}"
      for i in "${!options[@]}"; do
        if [[ "${answer,,}" == "${options[$i],,}" ]]; then
          printf '%s\n' "${options[$i]}"
          return 0
        fi
      done

      echo "Please enter a listed number." >&2
      sleep 1
      continue
    fi

    printf 'Select [1-%s]: ' "$count" >/dev/tty
    IFS= read -rsn1 key </dev/tty || true
    printf '\n' >/dev/tty
    if [[ "$key" == $'\x04' ]]; then
      return 1
    fi

    if [[ -z "$key" ]]; then
      printf '%s\n' "${options[$idx]}"
      return 0
    fi

    if [[ "$key" == $'\x1b' ]]; then
      read -rsn2 -t 0.1 key </dev/tty || true
      [[ -z "$key" ]] && return 1
      case "$key" in
        "[A") (( idx > 0 )) && idx=$((idx - 1)) ;;
        "[B") (( idx < count - 1 )) && idx=$((idx + 1)) ;;
      esac
      continue
    fi

    if [[ "$key" =~ ^[0-9]$ ]]; then
      selected_index="$key"
      if (( count >= 10 && key == 1 )); then
        read -rsn1 -t 0.25 answer </dev/tty || true
        if [[ "$answer" =~ ^[0-9]$ ]]; then
          selected_index="${key}${answer}"
        fi
      fi
      if (( selected_index >= 1 && selected_index <= count )); then
        printf '%s\n' "${options[$((selected_index - 1))]}"
        return 0
      fi
      echo "Please press a listed number." >&2
      sleep 1
      continue
    fi

    answer="$key"
    read -r -t 0.5 answer_rest </dev/tty || answer_rest=""
    answer="${answer}${answer_rest}"
    for i in "${!options[@]}"; do
      if [[ "${answer,,}" == "${options[$i],,}" ]]; then
        printf '%s\n' "${options[$i]}"
        return 0
      fi
    done
    echo "Please press a listed number." >&2
    sleep 1
  done
}

read_menu_key() {
  local max=$1
  local key next selected_index

  IFS= read -rsn1 key </dev/tty || return 1
  printf '\n' >/dev/tty
  [[ -n "$key" ]] || return 1
  [[ "$key" == $'\x04' ]] && return 1
  [[ "$key" == $'\x1b' ]] && return 1

  if [[ "$key" =~ ^[0-9]$ ]]; then
    selected_index="$key"
    if (( max >= 10 && key == 1 )); then
      read -rsn1 -t 0.25 next </dev/tty || true
      if [[ "$next" =~ ^[0-9]$ ]]; then
        selected_index="${key}${next}"
      fi
    fi
    if (( selected_index >= 0 && selected_index <= max )); then
      printf '%s\n' "$selected_index"
      return 0
    fi
  fi

  printf '%s\n' "$key"
}

read_line_with_esc() {
  local prompt=$1
  local key buffer=""

  printf '%s' "$prompt" >/dev/tty
  while true; do
    IFS= read -rsn1 key </dev/tty || return 130
    case "$key" in
      $'\x1b'|$'\x04')
        printf '\n' >/dev/tty
        return 130
        ;;
      "")
        printf '\n' >/dev/tty
        printf '%s\n' "$buffer"
        return 0
        ;;
      $'\x7f'|$'\b')
        if [[ -n "$buffer" ]]; then
          buffer=${buffer%?}
          printf '\b \b' >/dev/tty
        fi
        ;;
      *)
        buffer+="$key"
        printf '%s' "$key" >/dev/tty
        ;;
    esac
  done
}

prompt_default() {
  local label=$1
  local default=$2
  local answer

  if ! is_tty; then
    printf '%s\n' "$default"
    return 0
  fi

  answer=$(read_line_with_esc "$label [$default] (Esc to cancel): ") || return 130
  if [[ -z "$answer" ]]; then
    printf '%s\n' "$default"
  else
    printf '%s\n' "$answer"
  fi
}

prompt_required_dir() {
  local label=$1
  local default=$2
  local answer path

  while true; do
    if [[ -n "$default" ]]; then
      answer=$(read_line_with_esc "$label [$default] (Esc/q to cancel): ") || return 1
    else
      answer=$(read_line_with_esc "$label [required, Esc/q to cancel]: ") || return 1
    fi
    case "${answer,,}" in
      q|quit|exit)
        return 1
        ;;
    esac
    [[ -z "$answer" && -n "$default" ]] && answer="$default"
    [[ -z "$answer" ]] && return 1

    path="$answer"
    if [[ "$path" == "~" ]]; then
      path="$HOME"
    elif [[ "$path" == "~/"* ]]; then
      path="$HOME/${path#~/}"
    fi
    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
    echo "Directory does not exist: $path" >&2
    default="$path"
  done
}

prompt_checkpoint_dir() {
  local default=$1
  local answer path

  if ! is_tty; then
    printf '%s\n' "$default"
    return 0
  fi

  while true; do
    if [[ -n "$default" ]]; then
      answer=$(read_line_with_esc "Checkpoint directory [$default] (Esc/q to quit): ") || return 1
    else
      answer=$(read_line_with_esc "Checkpoint directory [required, Esc/q to quit]: ") || return 1
    fi

    case "${answer,,}" in
      q|quit|exit)
        echo "Start cancelled." >&2
        return 1
        ;;
    esac

    if [[ -z "$answer" && -n "$default" ]]; then
      answer="$default"
    fi
    if [[ -z "$answer" ]]; then
      echo "No checkpoint directory selected. Start cancelled." >&2
      return 1
    fi

    path="$answer"
    if [[ "$path" == "~" ]]; then
      path="$HOME"
    elif [[ "$path" == "~/"* ]]; then
      path="$HOME/${path#~/}"
    fi

    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi

    echo "Checkpoint directory does not exist: $path" >&2
    default="$path"
  done
}

prompt_choice() {
  local title=$1
  local current=$2
  shift
  shift
  menu_select "$title" "$current" "$@"
}

prompt_segmented() {
  local title=$1
  local current=$2
  shift
  shift
  local options=("$@")
  local key answer answer_rest i option

  while true; do
    printf '%s:\n' "$title" >/dev/tty
    for i in "${!options[@]}"; do
      option=${options[$i]}
      if [[ "$current" == "$option" ]]; then
        printf '  %d. [x] %s\n' "$((i + 1))" "$option" >/dev/tty
      else
        printf '  %d. [ ] %s\n' "$((i + 1))" "$option" >/dev/tty
      fi
    done
    printf 'Choose 1-%s, value, or Enter to keep: ' "${#options[@]}" >/dev/tty
    IFS= read -rsn1 key </dev/tty || true
    printf '\n' >/dev/tty
    [[ "$key" == $'\x1b' || "$key" == $'\x04' ]] && return 1
    case "${key,,}" in
      "" )
        printf '%s\n' "$current"
        return 0
        ;;
    esac

    if [[ "$key" =~ ^[0-9]$ ]] && (( key >= 1 && key <= ${#options[@]} )); then
      printf '%s\n' "${options[$((key - 1))]}"
      return 0
    fi

    if ((${#options[@]} == 2)); then
      case "${key,,}" in
        left|l)
          printf '%s\n' "${options[0]}"
          return 0
          ;;
        right|r)
          printf '%s\n' "${options[1]}"
          return 0
          ;;
      esac
    fi

    answer="$key"
    read -r -t 0.5 answer_rest </dev/tty || answer_rest=""
    answer="${answer}${answer_rest}"
    for option in "${options[@]}"; do
      if [[ "${answer,,}" == "${option,,}" ]]; then
        printf '%s\n' "$option"
        return 0
      fi
    done

    echo "Please choose a listed number, value, or Enter." >&2
  done
}

confirm_start() {
  local answer

  if ! is_tty; then
    return 0
  fi

  while true; do
    answer=$(read_line_with_esc "Start server now? [y/N]: ") || return 1
    case "$answer" in
      y|Y)
        return 0
        ;;
      n|N|"")
        echo "Start cancelled."
        return 1
        ;;
      *)
        echo "Please type y to start or n to exit."
        ;;
    esac
  done
}

show_help() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  cat <<'EOF'
This is the vLLM 2080 Ti Definitive service manager for a source checkout.

Main menu:
  1. Weight directory: choose the checkpoint directory.
  2. Profile: choose a profile directory, apply .env route presets, select a
     chat-template preset, and edit the filled runtime parameters.
  3. GPU / TP selection: select GPUs with Space; TP size follows GPU count.
  4. Launch mode: safe, normal, or fast.
  5. Port: default 8000.
  6. Service scope: local only or local + LAN.
  7. Help.
  8. Start service: launch vLLM, wait for /health, run a small smoke request,
     then print API URL, served model, PID file, and log file.
  9. Stop service.
  0. Exit.

Profiles are optional. They are presets only; the current menu values are the
actual launch configuration.

Notes:
  - safe mode is recommended for daily service: safe MTP sync, no forced full graph.
  - normal mode is the middle mode: nosync MTP, no forced full graph.
  - fast mode is the high-performance mode: nosync MTP with full graph enabled.
  - Chat-template presets live under profiles/templates and are global launcher
    settings, not route-profile fields.
  - thinking_token_budget is a per-request chat parameter in this vLLM runtime.
  - text+image requires a checkpoint that actually supports vision inputs.
  - --print-config prints the final launch summary and exits without starting.
EOF
  echo
  pause_enter
}

show_profiles() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Profile presets:"
  echo
  local profile profile_file family variant mode kv context mtp seqs group
  if [[ ! -d "$PROFILE_DIR" ]]; then
    echo "No profile directory found: $PROFILE_DIR"
    echo
    pause_enter
    return 0
  fi
  while IFS= read -r group; do
    [[ -n "$group" ]] || continue
    echo "=== $group ==="
    while IFS= read -r profile; do
      [[ -n "$profile" ]] || continue
      profile_file="$PROFILE_DIR/$group/$profile"
      family=$(read_profile_value "$profile_file" MODEL_FAMILY)
      variant=$(read_profile_value "$profile_file" MODEL_VARIANT)
      mode=$(read_profile_value "$profile_file" COMPATIBLE_MODES)
      [[ -n "$mode" ]] || mode=$(read_profile_value "$profile_file" MODE)
      kv=$(read_profile_value "$profile_file" KV_CACHE_DTYPE)
      context=$(read_profile_value "$profile_file" MAX_MODEL_LEN)
      mtp=$(read_profile_value "$profile_file" MTP_K)
      seqs=$(read_profile_value "$profile_file" MAX_NUM_SEQS)
      printf '    %-55s compatible=%-12s weight=%-6s kv=%-24s ctx=%-8s mtp=%-3s seqs=%s\n' \
        "$profile" "${mode:-safe,normal,fast}" "${variant:-auto}" "${kv:-fp16}" "${context:-auto}" "${mtp:-0}" "${seqs:-1}"
    done < <(list_profiles_in_group "$group")
    echo
  done < <(list_profile_groups)
  echo
  pause_enter
}

current_scope_label() {
  if [[ "${SERVICE_SCOPE:-local}" == "lan" ]]; then
    echo "local + LAN"
  else
    echo "local only"
  fi
}

current_profile_label() {
  if [[ -n "${PROFILE:-}" ]]; then
    echo "$PROFILE"
  else
    echo "none"
  fi
}

detect_default_gpu_devices() {
  local detected
  detected=$(
    list_nvidia_gpus 2>/dev/null |
      awk -F'\t' '
        BEGIN { sep = "" }
        tolower($2) ~ /2080[[:space:]]*ti/ {
          out = out sep $1
          sep = ","
          count++
          if (count == 2) {
            print out
            exit
          }
        }
      '
  ) || true
  if [[ -n "$detected" ]]; then
    printf '%s\n' "$detected"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '%s\n' "$CUDA_VISIBLE_DEVICES"
  else
    printf '0,1\n'
  fi
}

gpu_selected() {
  local idx=$1
  local devices=",${2// /},"
  [[ "$devices" == *",$idx,"* ]]
}

select_gpu_devices_menu() {
  local rows=() selected_devices idx=0 key count current_line gpu_idx gpu_name new_devices tp_count
  mapfile -t rows < <(list_nvidia_gpus || true)
  selected_devices=${GPU_DEVICES:-$(detect_default_gpu_devices)}

  if ((${#rows[@]} == 0)) || ! is_tty; then
    GPU_DEVICES=$(prompt_default "GPU devices / CUDA_VISIBLE_DEVICES" "$selected_devices") || return 0
    tp_count=$(gpu_device_count "$GPU_DEVICES")
    (( tp_count > 0 )) && TP_SIZE="$tp_count"
    save_manager_state
    return 0
  fi

  count=${#rows[@]}
  while true; do
    clear >/dev/tty
    {
      banner
      echo "GPU / TP selection"
      echo
      echo "Space toggles a GPU. Enter confirms. TP size follows selected GPU count."
      echo
      for i in "${!rows[@]}"; do
        current_line=${rows[$i]}
        gpu_idx=${current_line%%$'\t'*}
        gpu_name=${current_line#*$'\t'}
        if gpu_selected "$gpu_idx" "$selected_devices"; then
          mark="[x]"
        else
          mark="[ ]"
        fi
        if (( i == idx )); then
          printf ' > %s GPU %s  %s\n' "$mark" "$gpu_idx" "$gpu_name"
        else
          printf '   %s GPU %s  %s\n' "$mark" "$gpu_idx" "$gpu_name"
        fi
      done
      echo
      printf 'Selected: %s    TP_SIZE: %s\n' "${selected_devices:-none}" "$(gpu_device_count "$selected_devices")"
    } >/dev/tty

    IFS= read -rsn1 key </dev/tty || true
    if [[ "$key" == $'\x1b' ]]; then
      read -rsn2 -t 0.1 key </dev/tty || true
      [[ -z "$key" ]] && return 0
      case "$key" in
        "[A") (( idx > 0 )) && idx=$((idx - 1)) ;;
        "[B") (( idx < count - 1 )) && idx=$((idx + 1)) ;;
      esac
    elif [[ "$key" == " " ]]; then
      current_line=${rows[$idx]}
      gpu_idx=${current_line%%$'\t'*}
      if gpu_selected "$gpu_idx" "$selected_devices"; then
        new_devices=""
        IFS=',' read -r -a parts <<< "${selected_devices// /}"
        for part in "${parts[@]}"; do
          [[ -z "$part" || "$part" == "$gpu_idx" ]] && continue
          if [[ -n "$new_devices" ]]; then
            new_devices+=",$part"
          else
            new_devices="$part"
          fi
        done
        selected_devices="$new_devices"
      else
        if [[ -n "$selected_devices" ]]; then
          selected_devices+=",$gpu_idx"
        else
          selected_devices="$gpu_idx"
        fi
      fi
    elif [[ "$key" == "" ]]; then
      if [[ -z "$selected_devices" ]]; then
        echo "Select at least one GPU." >/dev/tty
        sleep 1
        continue
      fi
      GPU_DEVICES="$selected_devices"
      TP_SIZE=$(gpu_device_count "$GPU_DEVICES")
      save_manager_state
      return 0
    elif [[ "$key" == "q" || "$key" == "Q" ]]; then
      return 0
    fi
  done
}

discover_model_dirs() {
  local search_paths=${MODEL_SEARCH_PATHS:-/home/david/work/models}
  local dir
  IFS=':' read -r -a paths <<< "$search_paths"
  for dir in "${paths[@]}"; do
    [[ -d "$dir" ]] || continue
    find "$dir" -maxdepth 2 -mindepth 1 -type d \( -name '.cache' -o -name '.venv' -o -name '.*' \) -prune -o -type d -print 2>/dev/null |
      while IFS= read -r candidate; do
        if [[ -f "$candidate/config.json" ]] || ls "$candidate"/*.safetensors >/dev/null 2>&1; then
          printf '%s\n' "$candidate"
        fi
      done
  done | sort -u
}

select_weight_dir() {
  local dirs=() choices=() selected
  mapfile -t dirs < <(discover_model_dirs)
  if ((${#dirs[@]} == 0)); then
    selected=$(prompt_required_dir "Weight/checkpoint directory" "${MODEL_DIR:-}") || return 0
  else
    choices=("${dirs[@]}" "manual path" "change search path")
    selected=$(menu_select "Weight directory" "${MODEL_DIR:-${dirs[0]}}" "${choices[@]}") || return 0
    case "$selected" in
      "manual path")
        selected=$(prompt_required_dir "Weight/checkpoint directory" "${MODEL_DIR:-}") || return 0
        ;;
      "change search path")
        MODEL_SEARCH_PATHS=$(prompt_default "Search paths (colon-separated)" "${MODEL_SEARCH_PATHS:-/home/david/work/models}") || return 0
        save_manager_state
        select_weight_dir
        return 0
        ;;
    esac
  fi
  MODEL_DIR="$selected"
  MODEL_FAMILY=$(guess_model_family "$MODEL_DIR")
  QUANTIZATION=$(guess_quantization "$MODEL_DIR")
  SERVED_NAME=$(basename "$MODEL_DIR")
  save_manager_state
}

check_profile_model_match() {
  local profile_file=$1
  local profile_family current_family profile_group
  profile_family=$(read_profile_value "$profile_file" MODEL_FAMILY)
  profile_group=$(read_profile_value "$profile_file" PROFILE_GROUP)
  current_family=$(guess_model_family "${MODEL_DIR:-}")
  if [[ -n "$profile_family" && -n "$current_family" ]]; then
    if [[ "$profile_family" != "$current_family" ]]; then
      echo "WARNING: profile model family '$profile_family' does not match current model '$current_family'."
      echo
    fi
  fi
  if [[ -n "$profile_group" && -n "${MODEL_DIR:-}" ]]; then
    local model_basename
    model_basename=$(basename "$MODEL_DIR")
    if [[ -n "$profile_group" && "$profile_group" != *"${model_basename,,}"* && "${model_basename,,}" != *"${profile_group}"* ]]; then
      echo "NOTE: profile group '$profile_group' may not match model '$model_basename'."
      echo
    fi
  fi
}

apply_profile_preset_menu() {
  local groups=() profiles=() selected profile_file choices=() compatible_modes group
  mapfile -t groups < <(list_profile_groups)
  if ((${#groups[@]} == 0)); then
    mapfile -t profiles < <(list_profiles)
    if ((${#profiles[@]} == 0)); then
      echo "No .env profiles found under $PROFILE_DIR."
      echo
      pause_enter
      return 0
    fi
    choices=("Return" "${profiles[@]}")
    selected=$(menu_select "Profile preset" "${PROFILE:-Return}" "${choices[@]}") || return 0
    [[ "$selected" == "Return" ]] && return 0
    PROFILE="$selected"
  else
    choices=("Return" "${groups[@]}")
    group=$(menu_select "Model group" "Return" "${choices[@]}") || return 0
    [[ "$group" == "Return" ]] && return 0
    mapfile -t profiles < <(list_profiles_in_group "$group")
    if ((${#profiles[@]} == 0)); then
      echo "No profiles found under $group."
      echo
      pause_enter
      return 0
    fi
    choices=("Return" "${profiles[@]}")
    selected=$(menu_select "Profile [$group]" "Return" "${choices[@]}") || return 0
    [[ "$selected" == "Return" ]] && { apply_profile_preset_menu; return 0; }
    PROFILE="$group/$selected"
  fi
  profile_file="$PROFILE_DIR/$PROFILE"
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Applying profile preset: $PROFILE"
  echo
  profile_summary "$profile_file"
  echo
  check_profile_model_match "$profile_file"
  echo "Profile applied. Use \"Edit current runtime parameters\" if you want to override fields."
  echo
  apply_profile_overrides "$profile_file"
  compatible_modes=${COMPATIBLE_MODES:-$(read_profile_value "$profile_file" COMPATIBLE_MODES)}
  normalize_mode
  if ! mode_is_compatible "${MODE:-safe}" "$compatible_modes"; then
    MODE=$(first_compatible_mode "$compatible_modes")
    echo "Launch mode switched to compatible mode: $MODE"
    echo
  fi
  save_manager_state
  pause_enter
}

save_current_profile_menu() {
  local family_dir profile_name safe_name target_dir target_file answer compatible_modes

  if ! is_tty; then
    return 0
  fi

  family_dir=$(profile_family_dir)
  target_dir="$PROFILE_DIR/$family_dir/user"

  while true; do
    profile_name=$(read_line_with_esc "New profile name [${SERVED_NAME:-user-profile}] (Esc to cancel): ") || return 0
    [[ -z "$profile_name" ]] && profile_name=${SERVED_NAME:-user-profile}
    safe_name=$(sanitize_profile_name "$profile_name")
    target_file="$target_dir/${safe_name}.env"
    if [[ -e "$target_file" ]]; then
      answer=$(read_line_with_esc "Profile exists. Overwrite $target_file? [y/N]: ") || return 0
      case "$answer" in
        y|Y) break ;;
        *) continue ;;
      esac
    else
      break
    fi
  done

  mkdir -p "$target_dir"
  compatible_modes=$(profile_compatible_modes_for_current)
  : > "$target_file.tmp"
  write_profile_entry "$target_file.tmp" SERVED_NAME "${SERVED_NAME:-$safe_name}"
  write_profile_entry "$target_file.tmp" COMPATIBLE_MODES "$compatible_modes"
  write_profile_entry "$target_file.tmp" MODEL_FAMILY "${MODEL_FAMILY:-}"
  write_profile_entry "$target_file.tmp" PROFILE_GROUP "${PROFILE_GROUP:-}"
  write_profile_entry "$target_file.tmp" MODEL_VARIANT "${MODEL_VARIANT:-}"
  write_profile_entry "$target_file.tmp" QUANTIZATION "${QUANTIZATION:-}"
  write_profile_entry "$target_file.tmp" KV_CACHE_DTYPE "${KV_CACHE_DTYPE:-}"
  write_profile_entry "$target_file.tmp" MAX_MODEL_LEN "${MAX_MODEL_LEN:-}"
  write_profile_entry "$target_file.tmp" GPU_UTIL "${GPU_UTIL:-}"
  write_profile_entry "$target_file.tmp" MAX_BATCHED_TOKENS "${MAX_BATCHED_TOKENS:-}"
  write_profile_entry "$target_file.tmp" MAX_NUM_SEQS "${MAX_NUM_SEQS:-}"
  write_profile_entry "$target_file.tmp" MTP_K "${MTP_K:-}"
  write_profile_entry "$target_file.tmp" MESSAGE_TYPE "${MESSAGE_TYPE:-}"
  write_profile_entry "$target_file.tmp" MM_LIMIT_JSON "${MM_LIMIT_JSON:-}"
  write_profile_entry "$target_file.tmp" LANGUAGE_MODEL_ONLY "${LANGUAGE_MODEL_ONLY:-}"
  write_profile_entry "$target_file.tmp" SKIP_MM_PROFILING "${SKIP_MM_PROFILING:-}"
  write_profile_entry "$target_file.tmp" HF_OVERRIDES_JSON "${HF_OVERRIDES_JSON:-}"
  write_profile_entry "$target_file.tmp" ADDITIONAL_CONFIG_JSON "${ADDITIONAL_CONFIG_JSON:-}"
  write_profile_entry "$target_file.tmp" SPECULATIVE_CONFIG "${SPECULATIVE_CONFIG:-}"
  write_profile_entry "$target_file.tmp" COMPILATION_CONFIG_JSON "${COMPILATION_CONFIG_JSON:-}"
  write_profile_entry "$target_file.tmp" ATTENTION_BACKEND "${ATTENTION_BACKEND:-}"
  write_profile_entry "$target_file.tmp" ENFORCE_EAGER "${ENFORCE_EAGER:-}"
  write_profile_entry "$target_file.tmp" NO_ASYNC_SCHEDULING "${NO_ASYNC_SCHEDULING:-}"
  write_profile_entry "$target_file.tmp" DISABLE_HYBRID_KV_CACHE_MANAGER "${DISABLE_HYBRID_KV_CACHE_MANAGER:-}"
  write_profile_entry "$target_file.tmp" DISABLE_PREFIX_CACHING "${DISABLE_PREFIX_CACHING:-}"
  write_profile_entry "$target_file.tmp" DISABLE_CUSTOM_ALL_REDUCE "${DISABLE_CUSTOM_ALL_REDUCE:-}"
  write_profile_entry "$target_file.tmp" DISABLE_LOG_STATS "${DISABLE_LOG_STATS:-}"
  write_profile_entry "$target_file.tmp" ENABLE_TOOL_CALLING "${ENABLE_TOOL_CALLING:-}"
  write_profile_entry "$target_file.tmp" TOOL_CALL_PARSER "${TOOL_CALL_PARSER:-}"
  write_profile_entry "$target_file.tmp" VLLM_ENGINE_READY_TIMEOUT_S "${VLLM_ENGINE_READY_TIMEOUT_S:-}"
  write_profile_entry "$target_file.tmp" OMP_NUM_THREADS "${OMP_NUM_THREADS:-}"
  mv "$target_file.tmp" "$target_file"

  PROFILE="$family_dir/user/${safe_name}.env"
  save_manager_state
  echo "Saved profile: $target_file"
  echo "Compatible mode: $compatible_modes"
  echo
  pause_enter
}

change_profile_dir_menu() {
  local selected_dir
  selected_dir=$(prompt_required_dir "Profile directory" "${PROFILE_DIR:-$MANAGER_ROOT/profiles}") || return 0
  PROFILE_DIR="$selected_dir"
  TEMPLATE_DIR="$PROFILE_DIR/templates"
  save_manager_state
}

change_template_dir_menu() {
  local selected_dir
  selected_dir=$(prompt_required_dir "Template directory" "${TEMPLATE_DIR:-$PROFILE_DIR/templates}") || return 0
  TEMPLATE_DIR="$selected_dir"
  save_manager_state
}

select_template_preset_menu() {
  local templates=() choices=() selected resolved
  mapfile -t templates < <(list_template_presets)
  choices=("model default")
  if ((${#templates[@]} > 0)); then
    choices+=("${templates[@]}")
  fi
  choices+=("manual path" "change template directory" "Return")

  selected=$(menu_select "Chat template preset" "$(current_template_label)" "${choices[@]}") || return 0
  case "$selected" in
    "model default")
      CHAT_TEMPLATE_PRESET=""
      CHAT_TEMPLATE_FILE=""
      save_manager_state
      ;;
    "manual path")
      CHAT_TEMPLATE_FILE=$(prompt_optional "Chat template file" "${CHAT_TEMPLATE_FILE:-}") || return 0
      CHAT_TEMPLATE_PRESET=""
      save_manager_state
      ;;
    "change template directory")
      change_template_dir_menu
      ;;
    "Return")
      return 0
      ;;
    *)
      if resolved=$(resolve_template_file "$selected"); then
        CHAT_TEMPLATE_PRESET="$selected"
        CHAT_TEMPLATE_FILE="$resolved"
        save_manager_state
      else
        echo "Template preset not found: $selected"
        echo
        pause_enter
      fi
      ;;
  esac
}

edit_reasoning_defaults_menu() {
  local selected choices=()
  choices=(
    "template default"
    "thinking on"
    "thinking off"
    "custom kwargs"
    "reasoning parser"
    "default thinking budget"
    "Return"
  )
  selected=$(menu_select "Reasoning defaults" "$(current_reasoning_label)" "${choices[@]}") || return 0
  case "$selected" in
    "template default")
      REASONING_MODE=""
      DEFAULT_CHAT_TEMPLATE_KWARGS=""
      save_manager_state
      ;;
    "thinking on")
      REASONING_MODE=on
      DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
      save_manager_state
      ;;
    "thinking off")
      REASONING_MODE=off
      DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
      save_manager_state
      ;;
    "custom kwargs")
      DEFAULT_CHAT_TEMPLATE_KWARGS=$(prompt_optional "Default chat template kwargs JSON" "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}") || return 0
      REASONING_MODE=custom
      save_manager_state
      ;;
    "reasoning parser")
      REASONING_PARSER=$(prompt_optional "Reasoning parser" "${REASONING_PARSER:-}") || return 0
      save_manager_state
      ;;
    "default thinking budget")
      REASONING_BUDGET=$(prompt_optional "Default thinking token budget" "${REASONING_BUDGET:-}") || return 0
      echo "Default budget is applied only when a chat request does not provide thinking_token_budget."
      echo
      save_manager_state
      pause_enter
      ;;
    "Return")
      return 0
      ;;
  esac
}

select_profile_preset() {
  local selected choices=()

  while true; do
    if is_tty; then
      clear >/dev/tty 2>/dev/null || true
    fi
    banner
    echo "Profile manager"
    echo
    echo "Profiles are presets. Applying one fills the runtime parameters, then you"
    echo "can override every field before launching."
    echo
    printf '  Current profile dir: %s\n' "${PROFILE_DIR:-$MANAGER_ROOT/profiles}"
    printf '  Current profile:     %s\n' "$(current_profile_label)"
    printf '  Template dir:        %s\n' "${TEMPLATE_DIR:-$PROFILE_DIR/templates}"
    printf '  Chat template:       %s\n' "$(current_template_label)"
    printf '  Reasoning default:   %s\n' "$(current_reasoning_label)"
    echo

    choices=(
      "Apply profile preset"
      "Chat template preset"
      "Reasoning defaults"
      "Edit current runtime parameters"
      "Save current profile"
      "Change profile directory"
      "Clear profile"
      "Show profile list"
      "Return"
    )
    selected=$(menu_select "Profile action" "Apply profile preset" "${choices[@]}") || return 0
    case "$selected" in
      "Apply profile preset")
        apply_profile_preset_menu
        ;;
      "Chat template preset")
        select_template_preset_menu
        ;;
      "Reasoning defaults")
        edit_reasoning_defaults_menu
        ;;
      "Edit current runtime parameters")
        runtime_parameter_menu
        ;;
      "Save current profile")
        save_current_profile_menu
        ;;
      "Change profile directory")
        change_profile_dir_menu
        ;;
      "Clear profile")
        PROFILE=""
        save_manager_state
        ;;
      "Show profile list")
        show_profiles
        ;;
      "Return")
        return 0
        ;;
    esac
  done
}

prompt_optional() {
  local label=$1
  local default=${2:-}
  local answer

  if ! is_tty; then
    printf '%s\n' "$default"
    return 0
  fi

  answer=$(read_line_with_esc "$label [${default:-empty}] (type '-' to clear, Esc to cancel): ") || return 130
  if [[ "$answer" == "-" ]]; then
    printf '\n'
  elif [[ -z "$answer" ]]; then
    printf '%s\n' "$default"
  else
    printf '%s\n' "$answer"
  fi
}

prompt_toggle01() {
  local label=$1
  local default=${2:-0}
  local answer

  while true; do
    answer=$(read_line_with_esc "$label [$default] (0/1, Enter to keep, Esc to cancel): ") || return 130
    case "$answer" in
      "")
        printf '%s\n' "$default"
        return 0
        ;;
      0|1)
        printf '%s\n' "$answer"
        return 0
        ;;
      *)
        echo "Please enter 0 or 1." >&2
        ;;
    esac
  done
}

edit_advanced_parameters() {
  local answer

  if ! is_tty; then
    return 0
  fi

  answer=$(read_line_with_esc "Edit advanced optional parameters? [y/N]: ") || return 0
  case "$answer" in
    y|Y) ;;
    *) return 0 ;;
  esac

  echo
  ATTENTION_BACKEND=$(prompt_optional "Attention backend" "${ATTENTION_BACKEND:-}") || return 0
  HF_OVERRIDES_JSON=$(prompt_optional "HF overrides JSON" "${HF_OVERRIDES_JSON:-}") || return 0
  ADDITIONAL_CONFIG_JSON=$(prompt_optional "Additional config JSON" "${ADDITIONAL_CONFIG_JSON:-}") || return 0
  SPECULATIVE_CONFIG=$(prompt_optional "Speculative config JSON" "${SPECULATIVE_CONFIG:-}") || return 0
  COMPILATION_CONFIG_JSON=$(prompt_optional "Compilation config JSON" "${COMPILATION_CONFIG_JSON:-}") || return 0
  MM_LIMIT_JSON=$(prompt_optional "Multimodal limit JSON" "${MM_LIMIT_JSON:-}") || return 0
  if [[ -n "${MM_LIMIT_JSON:-}" && "${MESSAGE_TYPE:-text-only}" == "text-only" ]]; then
    MESSAGE_TYPE=text+image
    LANGUAGE_MODEL_ONLY=0
    SKIP_MM_PROFILING=${SKIP_MM_PROFILING:-0}
  fi
  LANGUAGE_MODEL_ONLY=$(prompt_toggle01 "Language-model only" "${LANGUAGE_MODEL_ONLY:-1}") || return 0
  SKIP_MM_PROFILING=$(prompt_toggle01 "Skip multimodal profiling" "${SKIP_MM_PROFILING:-1}") || return 0
  ENFORCE_EAGER=$(prompt_toggle01 "Enforce eager" "${ENFORCE_EAGER:-1}") || return 0
  NO_ASYNC_SCHEDULING=$(prompt_toggle01 "No async scheduling" "${NO_ASYNC_SCHEDULING:-0}") || return 0
  DISABLE_HYBRID_KV_CACHE_MANAGER=$(prompt_toggle01 "Disable hybrid KV cache manager" "${DISABLE_HYBRID_KV_CACHE_MANAGER:-0}") || return 0
  DISABLE_PREFIX_CACHING=$(prompt_toggle01 "Disable prefix caching" "${DISABLE_PREFIX_CACHING:-0}") || return 0
  DISABLE_CUSTOM_ALL_REDUCE=$(prompt_toggle01 "Disable custom all-reduce" "${DISABLE_CUSTOM_ALL_REDUCE:-0}") || return 0
  DISABLE_LOG_STATS=$(prompt_toggle01 "Disable log stats" "${DISABLE_LOG_STATS:-0}") || return 0
  ENABLE_TOOL_CALLING=$(prompt_toggle01 "Enable tool calling" "${ENABLE_TOOL_CALLING:-1}") || return 0
  if [[ "${ENABLE_TOOL_CALLING:-1}" == "1" ]]; then
    TOOL_CALL_PARSER=$(prompt_optional "Tool call parser (e.g. qwen3_xml, hermes)" "${TOOL_CALL_PARSER:-qwen3_xml}") || return 0
  fi
  VLLM_ENGINE_READY_TIMEOUT_S=$(prompt_optional "Engine ready timeout seconds" "${VLLM_ENGINE_READY_TIMEOUT_S:-1800}") || return 0
  OMP_NUM_THREADS=$(prompt_optional "OMP threads" "${OMP_NUM_THREADS:-8}") || return 0
}

edit_runtime_parameters() {
  local answer kv_choice message_choice current_message_type

  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
    banner
    echo "Runtime parameters"
    echo
    echo "Press Enter to keep the current value. Type '-' on optional fields to clear."
    echo
  fi

  MODEL_FAMILY=$(prompt_default "Model family" "${MODEL_FAMILY:-$(guess_model_family "${MODEL_DIR:-}")}") || return 0
  PROFILE_GROUP=$(prompt_optional "Profile group" "${PROFILE_GROUP:-}") || return 0
  MODEL_VARIANT=$(prompt_optional "Weight precision/profile variant" "${MODEL_VARIANT:-}") || return 0
  SERVED_NAME=$(prompt_default "Served model name" "${SERVED_NAME:-${MODEL_DIR:+$(basename "$MODEL_DIR")}}") || return 0
  QUANTIZATION=$(prompt_default "Weight quantization (empty/auto, fp8, gptq_marlin, awq_marlin, compressed-tensors)" "${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}") || return 0

  kv_choice=$(menu_select "KV precision" "${KV_CACHE_DTYPE:-fp16}" fp16 int8_per_token_head turboquant_k8v4 turboquant_4bit_nc) || return 0
  if [[ "$kv_choice" == "fp16" ]]; then
    KV_CACHE_DTYPE=""
  else
    KV_CACHE_DTYPE="$kv_choice"
  fi

  MAX_MODEL_LEN=$(prompt_default "Context tokens" "${MAX_MODEL_LEN:-$(default_context_tokens)}") || return 0
  GPU_UTIL=$(prompt_default "GPU memory utilization" "${GPU_UTIL:-$(default_gpu_util)}") || return 0
  MAX_BATCHED_TOKENS=$(prompt_default "Max batched tokens" "${MAX_BATCHED_TOKENS:-2048}") || return 0
  MAX_NUM_SEQS=$(prompt_default "Max concurrent sequences" "${MAX_NUM_SEQS:-1}") || return 0
  MTP_K=$(prompt_default "MTP speculative tokens" "${MTP_K:-0}") || return 0

  current_message_type=${MESSAGE_TYPE:-text-only}
  [[ -n "${MM_LIMIT_JSON:-}" ]] && current_message_type=text+image
  message_choice=$(menu_select "Message type" "$current_message_type" "text-only" "text+image") || return 0
  MESSAGE_TYPE="$message_choice"
  if [[ "$MESSAGE_TYPE" == "text+image" ]]; then
    MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    LANGUAGE_MODEL_ONLY=0
    SKIP_MM_PROFILING=${SKIP_MM_PROFILING:-0}
  else
    MM_LIMIT_JSON=""
    LANGUAGE_MODEL_ONLY=1
    SKIP_MM_PROFILING=1
  fi
  edit_advanced_parameters

  save_manager_state
}

edit_kv_precision_menu() {
  local kv_choice
  kv_choice=$(menu_select "KV precision" "${KV_CACHE_DTYPE:-fp16}" fp16 int8_per_token_head turboquant_k8v4 turboquant_4bit_nc) || return 0
  if [[ "$kv_choice" == "fp16" ]]; then
    KV_CACHE_DTYPE=""
  else
    KV_CACHE_DTYPE="$kv_choice"
  fi
  save_manager_state
}

edit_message_type_menu() {
  local current_message_type message_choice
  current_message_type=${MESSAGE_TYPE:-text-only}
  [[ -n "${MM_LIMIT_JSON:-}" ]] && current_message_type=text+image
  message_choice=$(menu_select "Message type" "$current_message_type" "text-only" "text+image") || return 0
  MESSAGE_TYPE="$message_choice"
  if [[ "$MESSAGE_TYPE" == "text+image" ]]; then
    MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    LANGUAGE_MODEL_ONLY=0
    SKIP_MM_PROFILING=${SKIP_MM_PROFILING:-0}
  else
    MM_LIMIT_JSON=""
    LANGUAGE_MODEL_ONLY=1
    SKIP_MM_PROFILING=1
  fi
  save_manager_state
}

runtime_parameter_menu() {
  local selected choices=()
  local model_family_value profile_group_value model_variant_value served_name_value
  local quantization_value kv_value context_value gpu_util_value
  local batch_tokens_value max_sequences_value mtp_value message_type_value
  local template_value reasoning_value tool_calling_value tool_parser_value engine_timeout_value omp_threads_value

  while true; do
    model_family_value=$(menu_value "${MODEL_FAMILY:-$(guess_model_family "${MODEL_DIR:-}")}")
    profile_group_value=$(menu_value "${PROFILE_GROUP:-}")
    model_variant_value=$(menu_value "${MODEL_VARIANT:-}")
    served_name_value=$(menu_value "${SERVED_NAME:-}")
    quantization_value=$(menu_value "${QUANTIZATION:-auto}")
    kv_value=$(menu_value "${KV_CACHE_DTYPE:-fp16}")
    context_value=$(menu_value "${MAX_MODEL_LEN:-$(default_context_tokens)}")
    gpu_util_value=$(menu_value "${GPU_UTIL:-$(default_gpu_util)}")
    batch_tokens_value=$(menu_value "${MAX_BATCHED_TOKENS:-2048}")
    max_sequences_value=$(menu_value "${MAX_NUM_SEQS:-1}")
    mtp_value=$(menu_value "${MTP_K:-0}")
    message_type_value=$(menu_value "${MESSAGE_TYPE:-text-only}")
    template_value=$(menu_value "$(current_template_label)")
    reasoning_value=$(menu_value "$(current_reasoning_label)")
    tool_calling_value=$(menu_value "${ENABLE_TOOL_CALLING:-0}")
    tool_parser_value=$(menu_value "${TOOL_CALL_PARSER:-}")
    engine_timeout_value=$(menu_value "${VLLM_ENGINE_READY_TIMEOUT_S:-}")
    omp_threads_value=$(menu_value "${OMP_NUM_THREADS:-}")

    if is_tty; then
      clear >/dev/tty 2>/dev/null || true
    fi
    banner
    echo "Runtime parameter overrides"
    echo
    choices=(
      "Model family: $model_family_value"
      "Profile group: $profile_group_value"
      "Weight variant: $model_variant_value"
      "Served name: $served_name_value"
      "Weight quantization: $quantization_value"
      "KV precision: $kv_value"
      "Context tokens: $context_value"
      "GPU util: $gpu_util_value"
      "Batch tokens: $batch_tokens_value"
      "Max sequences: $max_sequences_value"
      "MTP tokens: $mtp_value"
      "Message type: $message_type_value"
      "Chat template: $template_value"
      "Reasoning defaults: $reasoning_value"
      "Tool calling: $tool_calling_value"
      "Tool parser: $tool_parser_value"
      "Engine timeout: $engine_timeout_value"
      "OMP threads: $omp_threads_value"
      "Advanced options"
      "Edit all fields"
      "Return"
    )
    selected=$(menu_select "Runtime parameter" "Return" "${choices[@]}") || return 0
    case "$selected" in
      "Model family:"*)
        MODEL_FAMILY=$(prompt_default "Model family" "${MODEL_FAMILY:-$(guess_model_family "${MODEL_DIR:-}")}") || continue
        save_manager_state
        ;;
      "Profile group:"*)
        PROFILE_GROUP=$(prompt_optional "Profile group" "${PROFILE_GROUP:-}") || continue
        save_manager_state
        ;;
      "Weight variant:"*)
        MODEL_VARIANT=$(prompt_optional "Weight precision/profile variant" "${MODEL_VARIANT:-}") || continue
        save_manager_state
        ;;
      "Served name:"*)
        SERVED_NAME=$(prompt_default "Served model name" "${SERVED_NAME:-${MODEL_DIR:+$(basename "$MODEL_DIR")}}") || continue
        save_manager_state
        ;;
      "Weight quantization:"*)
        QUANTIZATION=$(prompt_default "Weight quantization (empty/auto, fp8, gptq_marlin, awq_marlin, compressed-tensors)" "${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}") || continue
        save_manager_state
        ;;
      "KV precision:"*)
        edit_kv_precision_menu
        ;;
      "Context tokens:"*)
        MAX_MODEL_LEN=$(prompt_default "Context tokens" "${MAX_MODEL_LEN:-$(default_context_tokens)}") || continue
        save_manager_state
        ;;
      "GPU util:"*)
        GPU_UTIL=$(prompt_default "GPU memory utilization" "${GPU_UTIL:-$(default_gpu_util)}") || continue
        save_manager_state
        ;;
      "Batch tokens:"*)
        MAX_BATCHED_TOKENS=$(prompt_default "Max batched tokens" "${MAX_BATCHED_TOKENS:-2048}") || continue
        save_manager_state
        ;;
      "Max sequences:"*)
        MAX_NUM_SEQS=$(prompt_default "Max concurrent sequences" "${MAX_NUM_SEQS:-1}") || continue
        save_manager_state
        ;;
      "MTP tokens:"*)
        MTP_K=$(prompt_default "MTP speculative tokens" "${MTP_K:-0}") || continue
        save_manager_state
        ;;
      "Message type:"*)
        edit_message_type_menu
        ;;
      "Chat template:"*)
        select_template_preset_menu
        ;;
      "Reasoning defaults:"*)
        edit_reasoning_defaults_menu
        ;;
      "Tool calling:"*)
        ENABLE_TOOL_CALLING=$(prompt_toggle01 "Enable tool calling" "${ENABLE_TOOL_CALLING:-1}") || continue
        if [[ "${ENABLE_TOOL_CALLING:-1}" == "1" && -z "${TOOL_CALL_PARSER:-}" ]]; then
          TOOL_CALL_PARSER=$(prompt_optional "Tool call parser (e.g. qwen3_xml, hermes)" "${TOOL_CALL_PARSER:-qwen3_xml}") || true
        fi
        save_manager_state
        ;;
      "Tool parser:"*)
        TOOL_CALL_PARSER=$(prompt_optional "Tool call parser (e.g. qwen3_xml, hermes)" "${TOOL_CALL_PARSER:-qwen3_xml}") || continue
        if [[ -n "${TOOL_CALL_PARSER:-}" ]]; then
          ENABLE_TOOL_CALLING=1
        fi
        save_manager_state
        ;;
      "Engine timeout:"*)
        VLLM_ENGINE_READY_TIMEOUT_S=$(prompt_optional "Engine ready timeout seconds (empty=default)" "${VLLM_ENGINE_READY_TIMEOUT_S:-1800}") || continue
        save_manager_state
        ;;
      "OMP threads:"*)
        OMP_NUM_THREADS=$(prompt_optional "OMP threads" "${OMP_NUM_THREADS:-8}") || continue
        save_manager_state
        ;;
      "Advanced options")
        edit_advanced_parameters
        save_manager_state
        ;;
      "Edit all fields")
        edit_runtime_parameters
        ;;
      "Return")
        return 0
        ;;
    esac
  done
}

normalize_mode() {
  # Compatibility shim for older state files or scripts.
  case "${MODE:-}" in
    stable)
      MODE=safe
      ;;
    speed)
      MODE=normal
      ;;
    aggressive)
      MODE=fast
      ;;
  esac
}

select_mode_menu() {
  normalize_mode
  MODE=$(prompt_segmented "Launch mode" "${MODE:-safe}" safe normal fast) || return 0
  save_manager_state
}

input_port_menu() {
  local current answer
  current=${PORT:-8000}

  if ! is_tty; then
    PORT="$current"
    save_manager_state
    return 0
  fi

  while true; do
    answer=$(read_line_with_esc "Port [$current] (q to cancel, Esc to return): ") || return 0
    case "${answer,,}" in
      q|quit|exit)
        return 0
        ;;
    esac
    [[ -z "$answer" ]] && answer="$current"
    if [[ "$answer" =~ ^[0-9]+$ ]] && (( answer >= 1 && answer <= 65535 )); then
      PORT="$answer"
      break
    fi
    echo "Please enter a port from 1 to 65535." >&2
  done
  save_manager_state
}

select_scope_menu() {
  local selected
  selected=$(prompt_segmented "Service scope" "$(current_scope_label)" "local only" "local + LAN") || return 0
  if [[ "$selected" == "local + LAN" ]]; then
    SERVICE_SCOPE=lan
  else
    SERVICE_SCOPE=local
  fi
  save_manager_state
}

show_status() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Service status:"
  echo
  mkdir -p "$LOG_DIR"
  local found=0 pid_file pid name state cmd
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    found=1
    name=$(basename "$pid_file" .pid)
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$pid" && -d "/proc/$pid" ]]; then
      state="running"
      cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | sed 's/[[:space:]]*$//')
    else
      state="stale"
      cmd="-"
    fi
    printf '  %-36s pid=%-8s %s\n' "$name" "${pid:-unknown}" "$state"
    [[ "$cmd" != "-" ]] && printf '    %s\n' "$cmd"
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)
  if (( found == 0 )); then
    echo "  No pid files found under $LOG_DIR."
  fi
  echo
  pause_enter
}

show_launch_status() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Service status"
  echo
  echo "  Status:       START OK"
  echo "  Served model: ${SERVED_NAME:-unknown}"
  echo "  Model path:   ${MODEL_DIR:-unknown}"
  echo "  GPU devices:  ${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-unknown}}"
  echo "  Mode:         ${MODE:-safe}"
  echo "  Scope:        ${SERVICE_SCOPE:-local}"
  echo "  Local API:    ${LAST_API_LOCAL:-http://127.0.0.1:${PORT:-8000}/v1}"
  if [[ -n "${LAST_API_LAN:-}" ]]; then
    echo "  LAN API:      $LAST_API_LAN"
  fi
  echo "  PID file:     ${LAST_PID_FILE:-unknown}"
  echo "  Log file:     ${LAST_LOG_FILE:-unknown}"
  if [[ -n "${LAST_SMOKE_OUTPUT:-}" ]]; then
    echo "  Smoke:        $LAST_SMOKE_OUTPUT"
  fi
}

print_running_services() {
  local pid_file pid name
  [[ -d "$LOG_DIR" ]] || return 0
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    pid=$(cat "$pid_file" 2>/dev/null || true)
    pid_is_running "$pid" || continue
    name=$(pid_file_service_name "$pid_file")
    printf '  %s pid=%s\n' "$name" "$pid"
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)
}

clear_last_service_state() {
  LAST_PID_FILE=""
  LAST_LOG_FILE=""
  LAST_API_LOCAL=""
  LAST_API_LAN=""
  LAST_SMOKE_OUTPUT=""
  save_manager_state
}

stop_pid_file() {
  local pid_file=$1
  local pid
  pid=$(cat "$pid_file" 2>/dev/null || true)
  if ! pid_is_running "$pid"; then
    rm -f "$pid_file"
    return 0
  fi

  kill "$pid" 2>/dev/null || true
  for _ in {1..20}; do
    pid_is_running "$pid" || {
      rm -f "$pid_file"
      return 0
    }
    sleep 0.5
  done

  return 1
}

stop_all_managed_services() {
  local pid_file failed=0
  [[ -d "$LOG_DIR" ]] || return 0
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    stop_pid_file "$pid_file" || failed=1
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)
  return "$failed"
}

confirm_restart_existing() {
  local answer

  current_service_info >/dev/null || return 0
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "A managed vLLM service is already running:"
  print_running_services
  echo
  answer=$(read_line_with_esc "Restart it with the current profile/config? [y/N]: ") || return 1
  case "$answer" in
    y|Y)
      echo "Stopping existing managed service..."
      stop_all_managed_services || {
        echo "Existing service did not stop cleanly. Use item 9 to inspect/stop it."
        return 1
      }
      clear_last_service_state
      echo "Existing service stopped."
      return 0
      ;;
    *)
      echo "Restart cancelled."
      return 1
      ;;
  esac
}

show_logs() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Recent logs:"
  echo
  mkdir -p "$LOG_DIR"
  local logs=() selected log_file
  mapfile -t logs < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.log' -printf '%T@ %f\n' 2>/dev/null | sort -nr | awk '{print $2}' | head -n 20)
  if ((${#logs[@]} == 0)); then
    echo "No logs found under $LOG_DIR."
    echo
    pause_enter
    return 0
  fi
  selected=$(menu_select "Log file" "${logs[0]}" "${logs[@]}") || return 0
  log_file="$LOG_DIR/$selected"
  clear >/dev/tty 2>/dev/null || true
  banner
  echo "Log: $log_file"
  echo
  tail -n "${LOG_TAIL_LINES:-120}" "$log_file" || true
  echo
  pause_enter
}

stop_service() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  mkdir -p "$LOG_DIR"
  local pid_files=() choices=() pid_file selected pid answer
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$pid" && -d "/proc/$pid" ]]; then
      pid_files+=("$pid_file")
      choices+=("$(basename "$pid_file" .pid) pid=$pid")
    fi
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)
  if ((${#choices[@]} == 0)); then
    echo "No running services found from $LOG_DIR pid files."
    echo
    pause_enter
    return 0
  fi
  selected=$(menu_select "Stop service" "${choices[0]}" "${choices[@]}") || return 0
  local index=-1
  for i in "${!choices[@]}"; do
    if [[ "${choices[$i]}" == "$selected" ]]; then
      index=$i
      break
    fi
  done
  (( index >= 0 )) || return 0
  pid_file="${pid_files[$index]}"
  pid=$(cat "$pid_file" 2>/dev/null || true)
  if [[ -z "$pid" || ! -d "/proc/$pid" ]]; then
    echo "Service is no longer running."
    rm -f "$pid_file"
    pause_enter
    return 0
  fi
  answer=$(read_line_with_esc "Stop $selected? [y/N]: ") || {
    echo "Stop cancelled."
    echo
    pause_enter
    return 0
  }
  case "$answer" in
    y|Y)
      if stop_pid_file "$pid_file"; then
        if [[ "${LAST_PID_FILE:-}" == "$pid_file" ]]; then
          clear_last_service_state
        fi
        echo "Stopped."
      else
        echo "Stop requested, but the process is still running."
      fi
      ;;
    *)
      echo "Stop cancelled."
      ;;
  esac
  echo
  pause_enter
}

guess_model_family() {
  local dir=${1,,}
  if [[ "$dir" == *gemma* ]]; then
    echo gemma4
  else
    echo qwen
  fi
}

guess_quantization() {
  local dir=${1,,}
  if [[ "$dir" == *fp8* ]]; then
    echo fp8
  elif [[ "$dir" == *gptq* ]]; then
    echo gptq_marlin
  elif [[ "$dir" == *awq* ]]; then
    echo awq_marlin
  else
    echo ""
  fi
}

default_context_tokens() {
  local quantization=${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}
  if [[ "$quantization" == "fp8" ]]; then
    echo 102400
  else
    echo 131072
  fi
}

default_gpu_util() {
  local quantization=${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}
  if [[ "$quantization" == "fp8" ]]; then
    echo 0.92
  else
    echo 0.90
  fi
}

apply_mode() {
  normalize_mode
  case "$MODE" in
    normal)
      export DISABLE_LOG_STATS=${DISABLE_LOG_STATS:-1}
      export VLLM_SM75_SPEC_SYNC_MODE=nosync
      export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0
      ;;
    fast)
      export DISABLE_LOG_STATS=${DISABLE_LOG_STATS:-1}
      export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE_OVERRIDE:-nosync}
      export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1
      ;;
    safe)
      export VLLM_SM75_SPEC_SYNC_MODE=safe
      export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0
      ;;
    *)
      die "MODE must be safe, normal, or fast."
      ;;
  esac
}

validate_mode_kv_policy() {
  local kv=${KV_CACHE_DTYPE:-}
  local mtp=${MTP_K:-0}
  local compatible_modes=${COMPATIBLE_MODES:-safe,normal,fast}
  local mode_ok=0
  local candidate
  normalize_mode
  local normalized_modes=${compatible_modes//,/ }
  for candidate in $normalized_modes; do
    case "$candidate" in
      stable)
        candidate=safe
        ;;
      speed)
        candidate=normal
        ;;
      aggressive)
        candidate=fast
        ;;
    esac
    if [[ "$candidate" == "$MODE" ]]; then
      mode_ok=1
      break
    fi
  done
  if (( mode_ok == 0 )); then
    echo "ERROR: MODE=$MODE is not compatible with this profile." >&2
    echo "       Compatible modes: $compatible_modes" >&2
    return 1
  fi
  case "$MODE" in
    safe)
      case "$kv" in
        ""|fp16|default|auto)
          ;;
        *)
          if [[ "$mtp" =~ ^[0-9]+$ ]] && (( mtp > 0 )); then
            echo "ERROR: safe mode allows quantized KV only when MTP_K=0." >&2
            echo "       Use MODE=normal or MODE=fast for quantized KV with MTP, or set MTP_K=0. KV: $kv" >&2
            return 1
          fi
          ;;
      esac
      ;;
    normal|fast)
      ;;
    *)
        echo "ERROR: MODE must be safe, normal, or fast." >&2
        return 1
        ;;
  esac
}

set_sm75_runtime_env() {
  local flashqla_candidate runtime_parent
  export STABLE_ROOT="$RUNTIME_ROOT"
  export HOME=${RUN_HOME:-"$HOME"}
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
  if [[ ! -x "$CUDA_HOME/bin/nvcc" && -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  fi
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    # Fall back to conda environment nvcc
    local conda_nvcc
    conda_nvcc=$(command -v nvcc 2>/dev/null || true)
    if [[ -n "$conda_nvcc" ]]; then
      export CUDA_HOME=$(dirname "$(dirname "$conda_nvcc")")
    fi
  fi
  export CUDA_PATH="$CUDA_HOME"
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.5}
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-$(detect_default_gpu_devices)}}"
  export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
  runtime_parent=$(cd -- "$RUNTIME_ROOT/.." && pwd)
  if [[ -z "${FLASHQLA_ROOT:-}" ]]; then
    for flashqla_candidate in \
      "$RUNTIME_ROOT/FlashQLA-SM70-SM75" \
      "$MANAGER_ROOT/FlashQLA-SM70-SM75" \
      "$runtime_parent/FlashQLA-SM70-SM75" \
      /opt/FlashQLA-SM70-SM75; do
      if [[ -d "$flashqla_candidate/flash_qla" ]]; then
        FLASHQLA_ROOT="$flashqla_candidate"
        break
      fi
    done
  fi
  export PYTHONPATH="$RUNTIME_ROOT${FLASHQLA_ROOT:+:$FLASHQLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
  export PATH="$RUNTIME_ROOT/.venv/bin:${CUDA_HOME}/bin:$PATH"
  export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
  export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
  # Keep generated kernels inside this runtime tree. Reusing cache dirs from
  # experiment worktrees can leave absolute paths to deleted environments.
  export TORCHINDUCTOR_CACHE_DIR="$MANAGER_ROOT/torchinductor-cache"
  export TRITON_CACHE_DIR="$MANAGER_ROOT/triton-cache"
  export PYTHONUNBUFFERED=1
  if [[ -n "${REASONING_BUDGET:-}" ]]; then
    export VLLM_DEFAULT_THINKING_TOKEN_BUDGET="$REASONING_BUDGET"
  else
    unset VLLM_DEFAULT_THINKING_TOKEN_BUDGET
  fi
  if [[ -n "${VLLM_ENGINE_READY_TIMEOUT_S:-}" ]]; then
    export VLLM_ENGINE_READY_TIMEOUT_S
  fi
  if [[ -n "${OMP_NUM_THREADS:-}" ]]; then
    export OMP_NUM_THREADS
  fi
}

build_args() {
  local host_arg=$1

  VLLM_ARGS=(
    --host "$host_arg"
    --port "$PORT"
    --model "$MODEL_DIR"
    --served-model-name "$SERVED_NAME"
    --dtype half
    --tensor-parallel-size "${TP_SIZE:-2}"
    --generation-config vllm
    --gpu-memory-utilization "$GPU_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --enable-chunked-prefill
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  )

  [[ -n "${QUANTIZATION:-}" ]] && VLLM_ARGS+=(--quantization "$QUANTIZATION")
  if [[ -n "${KV_CACHE_DTYPE:-}" && "${KV_CACHE_DTYPE}" != "fp16" ]]; then
    VLLM_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  fi
  [[ "${ENFORCE_EAGER:-0}" == "1" ]] && VLLM_ARGS+=(--enforce-eager)
  [[ "${NO_ASYNC_SCHEDULING:-0}" == "1" ]] && VLLM_ARGS+=(--no-async-scheduling)
  [[ "${DISABLE_HYBRID_KV_CACHE_MANAGER:-0}" == "1" ]] && VLLM_ARGS+=(--disable-hybrid-kv-cache-manager)
  if [[ "${DISABLE_PREFIX_CACHING:-0}" == "1" ]]; then
    VLLM_ARGS+=(--no-enable-prefix-caching)
  else
    VLLM_ARGS+=(--enable-prefix-caching)
  fi
  [[ "${LANGUAGE_MODEL_ONLY:-0}" == "1" ]] && VLLM_ARGS+=(--language-model-only)
  [[ "${SKIP_MM_PROFILING:-0}" == "1" ]] && VLLM_ARGS+=(--skip-mm-profiling)
  [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]] && VLLM_ARGS+=(--disable-custom-all-reduce)
  [[ "${DISABLE_LOG_STATS:-0}" == "1" ]] && VLLM_ARGS+=(--disable-log-stats)
  [[ -n "${ATTENTION_BACKEND:-}" ]] && VLLM_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
  [[ -n "${REASONING_PARSER:-}" ]] && VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
  [[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]] && VLLM_ARGS+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")
  [[ -n "${ADDITIONAL_CONFIG_JSON:-}" ]] && VLLM_ARGS+=(--additional-config "$ADDITIONAL_CONFIG_JSON")
  [[ -n "${HF_OVERRIDES_JSON:-}" ]] && VLLM_ARGS+=(--hf-overrides "$HF_OVERRIDES_JSON")
  [[ "${ENABLE_TOOL_CALLING:-0}" == "1" ]] && VLLM_ARGS+=(--enable-auto-tool-choice)
  [[ -n "${TOOL_CALL_PARSER:-}" ]] && VLLM_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")

  if [[ -n "${MM_LIMIT_JSON:-}" ]]; then
    VLLM_ARGS+=(--limit-mm-per-prompt "$MM_LIMIT_JSON")
  elif [[ "$MODEL_FAMILY" == qwen* ]]; then
    VLLM_ARGS+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
  elif [[ "$MODEL_FAMILY" == gemma* ]]; then
    VLLM_ARGS+=(--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}')
  fi

  if [[ "$MODEL_FAMILY" == qwen* && -n "${MM_LIMIT_JSON:-}" && -z "${ADDITIONAL_CONFIG_JSON:-}" ]]; then
    VLLM_ARGS+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
  fi

  if [[ -n "${CHAT_TEMPLATE_PRESET:-}" ]]; then
    local resolved_template
    if resolved_template=$(resolve_template_file "$CHAT_TEMPLATE_PRESET"); then
      CHAT_TEMPLATE_FILE="$resolved_template"
    fi
  fi
  [[ -n "${CHAT_TEMPLATE_FILE:-}" ]] && VLLM_ARGS+=(--chat-template "$CHAT_TEMPLATE_FILE")

  local capture=$((MTP_K + 1))
  if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
    VLLM_ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
  elif (( MTP_K > 0 )); then
    VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}")
  fi

  if [[ -n "${COMPILATION_CONFIG_JSON:-}" ]]; then
    VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG_JSON")
  elif [[ -n "${SPECULATIVE_CONFIG:-}" || "$MTP_K" -gt 0 ]]; then
    VLLM_ARGS+=(--compilation-config "{\"cudagraph_capture_sizes\":[${capture}],\"max_cudagraph_capture_size\":${capture}}")
  else
    VLLM_ARGS+=(--compilation-config '{"cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1}')
  fi
}

startup_status_line() {
  local log_file=$1
  [[ -s "$log_file" ]] || {
    printf 'waiting for first log line'
    return 0
  }
  tail -n 30 "$log_file" 2>/dev/null |
    awk '
      NF {
        line=$0
      }
      END {
        if (line == "") {
          exit
        }
        worker = ""
        if (match(line, /(worker_tp[0-9]+|Worker[^ :]*|TP[0-9]+)/)) {
          worker = substr(line, RSTART, RLENGTH)
        }
        sub(/^.*(INFO|WARNING|ERROR)[^:]*:[[:space:]]*/, "", line)
        if (worker != "") {
          print "(" worker " " line ")"
        } else {
          print line
        }
      }
    ' |
    tr '\t\r\n' '   ' |
    sed -E 's/[[:space:]]+/ /g; s/^[[:space:]]+|[[:space:]]+$//g' |
    cut -c1-100
}

startup_stage() {
  local log_file=$1

  if grep -qE 'Uvicorn running|Application startup complete|Started server process' "$log_file" 2>/dev/null; then
    printf 'Starting API server'
  elif grep -qE 'Capturing CUDA graphs|CUDA graph' "$log_file" 2>/dev/null; then
    printf 'Capturing CUDA graphs'
  elif grep -qE 'Loading model|Loading weights|model weights|safetensors|shard' "$log_file" 2>/dev/null; then
    printf 'Loading model weights'
  elif grep -qE 'Initializing|init|Engine' "$log_file" 2>/dev/null; then
    printf 'Initializing engine'
  else
    printf 'Starting process'
  fi
}

startup_progress_percent() {
  local elapsed=$1
  local timeout=$2
  local log_file=$3
  local percent

  if grep -qE 'Uvicorn running|Application startup complete' "$log_file" 2>/dev/null; then
    echo 95
    return 0
  fi
  if grep -qE 'Capturing CUDA graphs|CUDA graph' "$log_file" 2>/dev/null; then
    echo 75
    return 0
  fi
  if grep -qE 'Loading model|Loading weights|model weights|safetensors|shard' "$log_file" 2>/dev/null; then
    echo 45
    return 0
  fi
  percent=$((elapsed * 90 / timeout))
  (( percent < 5 )) && percent=5
  (( percent > 90 )) && percent=90
  echo "$percent"
}

progress_bar() {
  local percent=$1
  local width=${2:-32}
  local filled empty
  filled=$((percent * width / 100))
  empty=$((width - filled))
  printf '['
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '-'
  printf '] %3s%%' "$percent"
}

render_startup_progress() {
  local log_file=$1
  local elapsed=$2
  local timeout=$3
  local frame=$4
  local percent stage hint

  percent=$(startup_progress_percent "$elapsed" "$timeout" "$log_file")
  stage=$(startup_stage "$log_file")
  hint=$(startup_status_line "$log_file")

  {
    clear
    banner
    echo "Starting service"
    echo
    printf '  Progress: %s %s\n' "$(progress_bar "$percent")" "$frame"
    printf '  Time:     %ss / %ss\n' "$elapsed" "$timeout"
    printf '  Stage:    %s\n' "$stage"
    printf '  Status:   %s\n' "$hint"
    echo
    echo "Full startup log is still written to:"
    echo "  $log_file"
  } >/dev/tty
}

wait_for_ready() {
  local log_file=$1
  local url_host=$2
  local deadline=$((SECONDS + START_TIMEOUT))
  local fatal_regex='RuntimeError|ValueError|NotImplementedError|CUDA error|OutOfMemoryError|No supported config format|EngineCore encountered a fatal error|EngineDeadError'
  local pid=""
  local start_seconds=$SECONDS
  local frames=('|' '/' '-' '\\')
  local frame=0 hint elapsed last_notice=0
  if [[ -n "${CURRENT_SERVER_PID:-}" ]]; then
    pid="$CURRENT_SERVER_PID"
  fi

  while (( SECONDS < deadline )); do
    if curl -fsS "http://${url_host}:${PORT}/health" >/dev/null 2>&1; then
      if is_tty; then
        render_startup_progress "$log_file" "$((SECONDS - start_seconds))" "$START_TIMEOUT" "OK"
        printf 'Startup complete after %ss.\n' "$((SECONDS - start_seconds))" >/dev/tty
      fi
      return 0
    fi
    if grep -E "$fatal_regex" "$log_file" >/dev/null 2>&1; then
      if is_tty; then
        render_startup_progress "$log_file" "$((SECONDS - start_seconds))" "$START_TIMEOUT" "!"
        printf 'Startup failed.\n' >/dev/tty
      fi
      return 1
    fi
    if [[ -n "$pid" && ! -d "/proc/$pid" ]]; then
      if is_tty; then
        render_startup_progress "$log_file" "$((SECONDS - start_seconds))" "$START_TIMEOUT" "!"
        printf 'Startup failed: process exited.\n' >/dev/tty
      fi
      return 1
    fi
    elapsed=$((SECONDS - start_seconds))
    hint=$(startup_status_line "$log_file")
    if is_tty; then
      render_startup_progress "$log_file" "$elapsed" "$START_TIMEOUT" "${frames[$frame]}"
      frame=$(((frame + 1) % ${#frames[@]}))
    elif (( elapsed - last_notice >= 30 )); then
      echo "Starting server... elapsed=${elapsed}s | status=$hint"
      last_notice=$elapsed
    fi
    sleep 2
  done
  if is_tty; then
    render_startup_progress "$log_file" "$START_TIMEOUT" "$START_TIMEOUT" "!"
    printf 'Startup timed out after %ss.\n' "$START_TIMEOUT" >/dev/tty
  fi
  return 2
}

smoke_test() {
  local url_host=$1
  local model_id model_output
  model_output=$("$RUNTIME_ROOT/.venv/bin/python" - "$url_host" "$PORT" <<'PY'
import json
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=30) as resp:
    data = json.load(resp)
items = data.get("data") or []
if not items:
    raise SystemExit("no model returned")
print(items[0]["id"])
PY
)
  # Runtime sitecustomize hooks may print diagnostic lines on Python startup.
  model_id=$(printf '%s\n' "$model_output" | tail -n 1)

  "$RUNTIME_ROOT/.venv/bin/python" - "$url_host" "$PORT" "$model_id" <<'PY'
import json
import sys
import urllib.request

host, port, model_id = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "max_tokens": 64,
    "temperature": 0,
}
req = urllib.request.Request(
    f"http://{host}:{port}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.load(resp)
text = data["choices"][0]["message"].get("content", "")
if not text.strip():
    raise SystemExit("empty smoke response")
print(text.strip().replace("\n", " ")[:120])
PY
}

launch_server() {
  mkdir -p "$LOG_DIR"
  local safe_name log_file pid_file host_arg url_host args_text
  if [[ -z "${SERVED_NAME:-}" ]]; then
    SERVED_NAME=$(basename "$MODEL_DIR")
  fi
  GPU_DEVICES=${GPU_DEVICES:-$(detect_default_gpu_devices)}
  TP_SIZE=${TP_SIZE:-$(gpu_device_count "$GPU_DEVICES")}
  if [[ -z "${SERVED_NAME:-}" || "$SERVED_NAME" == "." || "$SERVED_NAME" == "/" ]]; then
    echo "ERROR: Served model name is empty. Set SERVED_NAME or choose a valid checkpoint directory." >&2
    return 1
  fi
  safe_name=$(printf '%s' "$SERVED_NAME" | tr -c 'A-Za-z0-9_.-' '_' | sed 's/_*$//')
  [[ -n "$safe_name" ]] || safe_name="vllm"
  log_file="$LOG_DIR/vllm-${safe_name}-${STAMP}.log"
  pid_file="$LOG_DIR/vllm-${safe_name}.pid"

  if [[ "$SERVICE_SCOPE" == "lan" ]]; then
    host_arg="0.0.0.0"
    url_host="127.0.0.1"
  else
    host_arg="127.0.0.1"
    url_host="127.0.0.1"
  fi

  build_args "$host_arg"
  printf -v args_text '%q ' "${VLLM_ARGS[@]}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo
    echo "DRY RUN"
    echo "Environment:"
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "  VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-auto}"
    echo "  VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-0}"
    echo "  VLLM_DEFAULT_THINKING_TOKEN_BUDGET=${VLLM_DEFAULT_THINKING_TOKEN_BUDGET:-}"
    echo "Command:"
    echo "  $RUNTIME_ROOT/.venv/bin/python -m vllm.entrypoints.openai.api_server $args_text"
    return 0
  fi

  echo
  echo "Starting server..."
  echo "  Log: $log_file"
  echo "  Mode: $MODE"
  echo "  MTP graph policy: VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-auto}, VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-0}"
  echo "  Served name: $SERVED_NAME"
  echo "  Model: $MODEL_DIR"
  echo "  Bind: $host_arg:$PORT"

  nohup "$RUNTIME_ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" >"$log_file" 2>&1 &
  CURRENT_SERVER_PID=$!
  echo "$CURRENT_SERVER_PID" > "$pid_file"

  local ready_rc=0
  wait_for_ready "$log_file" "$url_host" || ready_rc=$?
  if [[ "$ready_rc" != "0" ]]; then
    echo
    echo "START FAILED"
    echo "Log: $log_file"
    tail -n 120 "$log_file" || true
    kill "$(cat "$pid_file" 2>/dev/null)" >/dev/null 2>&1 || true
    rm -f "$pid_file"
    return 1
  fi

  echo "Health check: OK"
  local smoke_output
  echo "Running smoke test..."
  if ! smoke_output=$(smoke_test "$url_host" 2>&1); then
    echo
    echo "SMOKE FAILED"
    echo "$smoke_output"
    echo "Log: $log_file"
    kill "$(cat "$pid_file" 2>/dev/null)" >/dev/null 2>&1 || true
    rm -f "$pid_file"
    return 1
  fi

  local api_local="http://127.0.0.1:${PORT}/v1"
  local api_lan=""
  if [[ "$SERVICE_SCOPE" == "lan" ]]; then
    api_lan="http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}/v1"
  fi
  LAST_PID_FILE="$pid_file"
  LAST_LOG_FILE="$log_file"
  LAST_API_LOCAL="$api_local"
  LAST_API_LAN="$api_lan"
  LAST_SMOKE_OUTPUT="$smoke_output"
  save_manager_state

  echo
  echo "START OK"
  echo "Smoke response: $smoke_output"
  echo "PID file: $pid_file"
  echo "Log: $log_file"
  echo "Local API: $api_local"
  if [[ -n "$api_lan" ]]; then
    echo "LAN API:   $api_lan"
  fi

  if is_tty; then
    show_launch_status
  fi
}

prepare_runtime_defaults() {
  if [[ -z "${MODEL_DIR:-}" ]]; then
    echo "ERROR: MODEL_DIR is required. Choose item 1 first." >&2
    return 1
  fi
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory does not exist: $MODEL_DIR" >&2
    return 1
  fi
  MODEL_FAMILY=${MODEL_FAMILY:-$(guess_model_family "$MODEL_DIR")}
  SERVED_NAME=${SERVED_NAME:-$(basename "$MODEL_DIR")}
  TEMPLATE_DIR=${TEMPLATE_DIR:-"$PROFILE_DIR/templates"}
  GPU_DEVICES=${GPU_DEVICES:-$(detect_default_gpu_devices)}
  TP_SIZE=${TP_SIZE:-$(gpu_device_count "$GPU_DEVICES")}
  QUANTIZATION=${QUANTIZATION:-$(guess_quantization "$MODEL_DIR")}
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(default_context_tokens)}
  GPU_UTIL=${GPU_UTIL:-$(default_gpu_util)}
  MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-2048}
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
  MTP_K=${MTP_K:-0}
  PORT=${PORT:-8000}
  MODE=${MODE:-safe}
  normalize_mode
  SERVICE_SCOPE=${SERVICE_SCOPE:-local}
  if [[ -z "${MESSAGE_TYPE:-}" && -n "${MM_LIMIT_JSON:-}" ]]; then
    MESSAGE_TYPE=text+image
  else
    MESSAGE_TYPE=${MESSAGE_TYPE:-text-only}
  fi
  if [[ "$MESSAGE_TYPE" == "text+image" ]]; then
    MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-0}
    SKIP_MM_PROFILING=${SKIP_MM_PROFILING:-0}
  else
    MM_LIMIT_JSON=""
    LANGUAGE_MODEL_ONLY=1
    SKIP_MM_PROFILING=1
  fi
  ENFORCE_EAGER=${ENFORCE_EAGER:-1}
  ENABLE_TOOL_CALLING=${ENABLE_TOOL_CALLING:-1}
  TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_xml}
  VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
  validate_mode_kv_policy
}

collect_config_env() {
  local profile_file
  if profile_file=$(resolve_profile_file); then
    apply_profile_overrides "$profile_file"
  fi
  prepare_runtime_defaults || die "Invalid runtime configuration."
}

print_review() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  local message_type=text-only
  [[ -n "${MM_LIMIT_JSON:-}" ]] && message_type=text+image

  cat <<EOF
Launch summary:
  Model directory:      $MODEL_DIR
  Served name:          $SERVED_NAME
  Model family:         $MODEL_FAMILY
  Weight quantization:  ${QUANTIZATION:-auto}
  GPU devices:          ${GPU_DEVICES:-$(detect_default_gpu_devices)}
  KV precision:         ${KV_CACHE_DTYPE:-fp16}
  Context tokens:       $MAX_MODEL_LEN
  GPU util:             $GPU_UTIL
  Max batched tokens:   $MAX_BATCHED_TOKENS
  Max sequences:        $MAX_NUM_SEQS
  MTP tokens:           $MTP_K
  Message type:         $message_type
  Chat template:        $(current_template_label)
  Reasoning default:    $(current_reasoning_label)
  Tool calling:         ${ENABLE_TOOL_CALLING:-0}
  Tool parser:          ${TOOL_CALL_PARSER:-<none>}
  Engine timeout:       ${VLLM_ENGINE_READY_TIMEOUT_S:-default}
  OMP threads:          ${OMP_NUM_THREADS:-8}
  Mode:                 $MODE
  MTP graph policy:     VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-auto}, VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-0}
  Port:                 $PORT
  Scope:                $SERVICE_SCOPE
EOF
  echo
}

start_configured_service() {
  if [[ -z "${MODEL_DIR:-}" ]]; then
    echo "Set item 1: weight/checkpoint directory first."
    pause_enter
    return 0
  fi

  if ! prepare_runtime_defaults; then
    pause_enter
    return 0
  fi

  apply_mode
  set_sm75_runtime_env
  START_TIMEOUT=${START_TIMEOUT:-900}
  print_review
  if current_service_info >/dev/null; then
    confirm_restart_existing || {
      pause_enter
      return 0
    }
  else
    confirm_start || return 0
  fi

  if launch_server; then
    echo
    echo "Press Enter to return to the main menu. The service will keep running."
    pause_enter
  else
    echo
    echo "Returned to main menu."
    pause_enter
  fi
}

menu_value() {
  local value=${1:-}
  if [[ -n "$value" ]]; then
    printf '%s' "$value"
  else
    printf '<unset>'
  fi
}

render_main_menu_item() {
  local idx=$1
  local current=$2
  local text=$3

  if (( idx == current )); then
    printf ' > %s\n' "$text"
  else
    printf '   %s\n' "$text"
  fi
}

render_main_menu() {
  local current=${1:-1}
  local gpu_devices tp_size
  gpu_devices=${GPU_DEVICES:-$(detect_default_gpu_devices)}
  tp_size=${TP_SIZE:-$(gpu_device_count "$gpu_devices")}

  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Main menu"
  echo
  render_service_status
  render_main_menu_item 1 "$current" "1. Weight directory: ${MODEL_DIR:+$(basename "$MODEL_DIR")}${MODEL_DIR:-<unset>}"
  render_main_menu_item 2 "$current" "2. Profile:          $(current_profile_label)"
  printf '     Model family:     %s\n' "$(menu_value "${MODEL_FAMILY:-}")"
  printf '     Served name:      %s\n' "$(menu_value "${SERVED_NAME:-}")"
  printf '     Weight quant:     %s\n' "$(menu_value "${QUANTIZATION:-auto}")"
  printf '     KV precision:     %s\n' "$(menu_value "${KV_CACHE_DTYPE:-fp16}")"
  printf '     Context tokens:   %s\n' "$(menu_value "${MAX_MODEL_LEN:-$(default_context_tokens)}")"
  printf '     MTP tokens:       %s\n' "$(menu_value "${MTP_K:-0}")"
  printf '     Message type:     %s\n' "$(menu_value "${MESSAGE_TYPE:-text-only}")"
  printf '     [Runtime] GPU util=%s batch=%s seqs=%s template=%s reasoning=%s\n' \
    "${GPU_UTIL:-$(default_gpu_util)}" "${MAX_BATCHED_TOKENS:-2048}" "${MAX_NUM_SEQS:-1}" \
    "$(current_template_label)" "$(current_reasoning_label)"
  printf '     [Tools]   calling=%s parser=%s timeout=%s omp=%s\n' \
    "${ENABLE_TOOL_CALLING:-1}" "${TOOL_CALL_PARSER:-<none>}" \
    "${VLLM_ENGINE_READY_TIMEOUT_S:-1800}" "${OMP_NUM_THREADS:-8}"
  render_main_menu_item 3 "$current" "3. GPU/TP setting:  $(menu_value "$gpu_devices") / TP $(menu_value "$tp_size")"
  render_main_menu_item 4 "$current" "4. Launch mode:      ${MODE:-safe}"
  render_main_menu_item 5 "$current" "5. Port:             ${PORT:-8000}"
  render_main_menu_item 6 "$current" "6. Service scope:    $(current_scope_label)"
  render_main_menu_item 7 "$current" "7. Help"
  render_main_menu_item 8 "$current" "8. Start service"
  render_main_menu_item 9 "$current" "9. Stop service"
  render_main_menu_item 0 "$current" "0. Exit"
  echo
  echo "Use Up/Down, Enter to select. Number keys jump directly. Esc/0 exits."
  echo "Profile presets are optional. They only fill editable runtime parameters."
  echo
}

read_main_menu_choice() {
  local current=$1
  local key seq selected_index

  IFS= read -rsn1 key </dev/tty || return 1
  printf '\n' >/dev/tty

  [[ "$key" == $'\x04' ]] && return 1
  if [[ -z "$key" ]]; then
    printf '%s\n' "$current"
    return 0
  fi

  if [[ "$key" == $'\x1b' ]]; then
    read -rsn2 -t 0.1 seq </dev/tty || true
    if [[ -z "$seq" ]]; then
      return 1
    fi
    case "$seq" in
      "[A")
        if (( current > 1 )); then
          current=$((current - 1))
        else
          current=0
        fi
        printf '__INDEX__:%s\n' "$current"
        return 0
        ;;
      "[B")
        if (( current == 0 )); then
          current=1
        elif (( current < 9 )); then
          current=$((current + 1))
        else
          current=0
        fi
        printf '__INDEX__:%s\n' "$current"
        return 0
        ;;
    esac
    return 0
  fi

  if [[ "$key" =~ ^[0-9]$ ]]; then
    printf '%s\n' "$key"
    return 0
  fi

  selected_index="$key"
  printf '%s\n' "$selected_index"
}

service_manager() {
  local choice menu_idx=1
  load_manager_state
  MODE=${MODE:-safe}
  PORT=${PORT:-8000}
  SERVICE_SCOPE=${SERVICE_SCOPE:-local}

  while true; do
    render_main_menu "$menu_idx"
    printf 'Select [0-9]: ' >/dev/tty
    if ! choice=$(read_main_menu_choice "$menu_idx"); then
      exit 0
    fi
    case "$choice" in
      __INDEX__:*)
        menu_idx=${choice#__INDEX__:}
        continue
        ;;
    esac
    case "$choice" in
      1)
        select_weight_dir
        ;;
      2)
        select_profile_preset
        ;;
      3)
        select_gpu_devices_menu
        ;;
      4)
        select_mode_menu
        ;;
      5)
        input_port_menu
        ;;
      6)
        select_scope_menu
        ;;
      7)
        show_help
        ;;
      8)
        start_configured_service
        ;;
      9)
        stop_service
        ;;
      0|q|Q|quit|exit)
        exit 0
        ;;
      "")
        ;;
      *)
        echo "Unknown choice: $choice"
        pause_enter
        ;;
    esac
  done
}

has_arg() {
  local want=$1
  shift || true
  local arg
  for arg in "$@"; do
    [[ "$arg" == "$want" ]] && return 0
  done
  return 1
}

run_start_flow() {
  local profile_file
  if profile_file=$(resolve_profile_file); then
    apply_profile_overrides "$profile_file"
  fi
  collect_config_env
  apply_mode
  set_sm75_runtime_env
  START_TIMEOUT=${START_TIMEOUT:-900}
  print_review
  if [[ "${PRINT_CONFIG:-0}" == "1" ]] || has_arg "--print-config" "$@"; then
    return 0
  fi
  launch_server
}

main() {
  cd "$MANAGER_ROOT"

  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
    exit 0
  fi

  if [[ ! -x "$RUNTIME_ROOT/.venv/bin/python" ]]; then
    banner
    die ".venv is missing under RUNTIME_ROOT=$RUNTIME_ROOT. Run ./build.sh first or set RUNTIME_ROOT."
  fi

  mkdir -p "$LOG_DIR"

  if [[ "${1:-}" == "--non-interactive" || "${1:-}" == "--print-config" || "${NON_INTERACTIVE:-0}" == "1" || ! -t 0 ]]; then
    run_start_flow "$@"
  else
    service_manager
  fi
}

main "$@"
