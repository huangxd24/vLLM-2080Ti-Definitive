# Changelog

This changelog tracks the fork release version for vLLM 2080 Ti Definitive
Edition. It is separate from the upstream vLLM package version.

## v0.1.7 - 2026-06-13

- Explicitly passes `--enable-prefix-caching` by default so vLLM caches shared
  KV prefixes (system prompt, tool definitions) across Hermes multi-turn
  requests; set `DISABLE_PREFIX_CACHING=1` to opt out.
- Increases smoke-test `max_tokens` from 8 to 64 so the startup warmup covers
  more Triton decode-kernel shapes, reducing first-request JIT latency spikes.
- Adds user profile `qwen27b-fp8-fp16kv-64K-mtp3-text-only` with MTP_K=3
  and explicit `KV_CACHE_DTYPE=fp16` to properly override stale INT8 KV state
  from previous profiles when switching.
- Fixes `CUDA_HOME` detection to fall back to conda-env `nvcc` when
  `/usr/local/cuda-12.8` and `/usr/local/cuda` do not exist, preventing
  FlashInfer JIT compilation failures during MTP startup.
- Disables FlashInfer top-k/top-p sampler by default
  (`VLLM_USE_FLASHINFER_SAMPLER=0`) to avoid runtime JIT compilation timeouts
  on SM75 that crash the engine during MTP rejection sampling.
- Filters `fp16` from `--kv-cache-dtype` argument since vLLM CLI only accepts
  `float16`; the launcher now treats `fp16` and empty as "use default".
- Renames profile directory `qwen27b` to `qwopus36-27b` to align with model
  short name; updates all PROFILE_GROUP, SERVED_NAME, README, docs, and state
  file references.
- Adds weight directory auto-discovery: `select_weight_dir` now scans
  `MODEL_SEARCH_PATHS` for model directories and presents a list selector
  with manual-path fallback.
- Restructures profile selection into a two-level menu: model group first,
  then profiles within the chosen group.
- Adds profile-model consistency check that warns when a profile's model
  family or group does not match the current weight directory.
- Simplifies main menu layout: condenses runtime and tool settings into
  two summary lines, shows weight directory basename instead of full path.

## v0.1.6 - 2026-06-12

- Integrates `ENABLE_TOOL_CALLING`, `TOOL_CALL_PARSER`,
  `VLLM_ENGINE_READY_TIMEOUT_S`, and `OMP_NUM_THREADS` into the launcher menu
  system, state persistence, profile save/load, and runtime environment export.
- Sets production defaults matching README.zh-CN.md: `ENFORCE_EAGER=1`,
  `ENABLE_TOOL_CALLING=1`, `TOOL_CALL_PARSER=qwen3_xml`,
  `VLLM_ENGINE_READY_TIMEOUT_S=1800`, `OMP_NUM_THREADS=8`.
- Adds dedicated menu items (Tool calling, Tool parser, Engine timeout,
  OMP threads) to the runtime parameter overrides screen.
- Adds `Qwopus3.5-9B-Code` profile under
  `profiles/qwopus35-9b-code/safe/fp16/fp16kv-128K-nomtp-text-only.env`
  with FP16 KV, 128K context, text-only, and tool calling enabled.

## v0.1.5 - 2026-06-08

- Renames the public service manager to `launcher.sh` and keeps `build.sh` as
  the one-click source build entry point.
- Updates launcher modes to `safe`, `normal`, and `fast`, with route profiles
  split by model, mode, and weight precision.
- Adds chat template presets and service-level thinking budget defaults while
  keeping global runtime controls out of route profile files.
- Refreshes the Qwen3.6 profile documentation and restores the KV throughput
  sweep SVG charts.

## v0.1.4 - 2026-06-06

- Slims the public repository down to the focused SM75 runtime source tree,
  launcher scripts, validated profiles, and project documentation.
- Keeps Docker artifacts out of this source release; Docker packaging remains a
  separate future deployment layer.
- Adds the interactive `launcher.sh` service manager and one-click `build.sh`
  source build entry point.
- Uses the public launcher modes `safe`, `normal`, and `fast`, with validated
  profiles organized under model-specific profile directories.
- Carries forward the `v0.1.3` graph-safety runtime fixes while removing
  upstream CI/docs/test bulk from the public source tree.

## v0.1.3 - 2026-06-05

- Adds the issue #24 MTP graph-safety fix for hybrid Mamba/GDN models.
- Makes production profiles safer by default: Native MTP + hybrid recurrent KV
  layers fall back from full decode CUDA Graph replay to PIECEWISE/NONE.
- Keeps the old peak-throughput route available for explicit speed benchmarking
  via `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1`.

## v0.1.2 - 2026-06-04

- Public stable snapshot for the SM75 TP=2 CUDA 12.8 runtime.
- Keeps the upstream vLLM base at `0.21.0` while versioning this fork as an
  independent 2080 Ti runtime distribution.
- Updates the documented Qwen3.6 and Gemma4 runtime routes, tested checkpoint
  list, launcher profile guidance, and benchmark evidence links.

## v0.1.1

- Follow-up compatibility fixes for editable/source builds and optional CUDA
  extension imports on SM75 environments.

## v0.1.0

- Initial public stable snapshot of the dual 2080 Ti / SM75 TP=2 runtime.
