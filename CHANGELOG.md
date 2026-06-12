# Changelog

This changelog tracks the fork release version for vLLM 2080 Ti Definitive
Edition. It is separate from the upstream vLLM package version.

## v0.1.6 - 2026-06-12

- Integrates `ENABLE_TOOL_CALLING`, `TOOL_CALL_PARSER`,
  `VLLM_ENGINE_READY_TIMEOUT_S`, and `OMP_NUM_THREADS` into the launcher menu
  system, state persistence, profile save/load, and runtime environment export.
- Sets production defaults matching README.zh-CN.md: `ENFORCE_EAGER=1`,
  `ENABLE_TOOL_CALLING=1`, `TOOL_CALL_PARSER=qwen3_xml`,
  `VLLM_ENGINE_READY_TIMEOUT_S=1800`, `OMP_NUM_THREADS=8`.
- Adds dedicated menu items (Tool calling, Tool parser, Engine timeout,
  OMP threads) to the runtime parameter overrides screen.

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
