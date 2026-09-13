<!-- markdownlint-disable MD001 MD041 -->
# ⚡ vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition 题图](docs/assets/vllm-2080ti-cover.jpg)

面向双 RTX 2080 Ti / SM75 推理的终极版 vLLM 运行时。

这是一个硬件定向的 vLLM fork，用来保存已经跑通的 2080 Ti vLLM
栈：补丁源码、启动 profile、运行时说明和稳定环境记录。

Fork 发布版本：`v0.1.5`
基础 vLLM：`0.21.0`

核心实测：同一套双 2080 Ti TP=2 runtime 下，Qwen3.6 27B 单请求 decode
达到 `100+ tok/s`，Gemma4 31B 单请求 decode 达到 `~100 tok/s`。

语言：[English](README.md) | 简体中文

![单请求实时测速演示](docs/assets/vllmspeed.gif)

## 💡 为什么用 RTX 2080 Ti 做 LLM 推理？

2018 年 8 月，NVIDIA 推出了划时代的 RTX 2080 Ti 系列显卡，并将玩家
显卡产品线从 GTX 带入 RTX 时代，从此开启了实时光追时代。这一代显卡
给无数电脑爱好者留下了难以磨灭的印记。八年之后，2080 Ti 依然可以在
2K 分辨率下流畅运行当下主流 3A 大作，可以说是老骥伏枥，志在千里。

而当年的 2080 Ti 还留下了两个非常关键的硬件空间：一是可以把 11 颗
1GB GDDR6 显存颗粒升级为 2GB 容量，从而获得 22GB 可用显存；二是它
保留了在 40 系之后被消费级显卡淘汰的 NVLink 高速互联接口。当高规格
核心、改造后的大显存、高速卡间互联，以及以今天眼光看依然很快的显存
带宽叠加在一起，我们在审视本地 AI 推理时发现，这个组合仍然有巨大的
用武之地。具体而言：

| 指标 | 2x 2080 Ti 22GB + NVLink | 3090 Ti 24GB 基线 | 倍率 |
|---|---:|---:|---:|
| 物理 CUDA core 数量 | 8,704 | 5,376 | 1.62x |
| SM 数量 | 136 | 84 | 1.62x |
| 物理 Tensor Core 数量 | 1,088 | 336 | 3.24x |
| Dense Tensor FP16 matrix throughput | 228 TFLOPS | 160 TFLOPS | 1.43x |
| 总物理显存带宽 | 1,232 GB/s | 1,008 GB/s | 1.22x |
| 总显存容量 | 44GB | 24GB | 1.83x |
| 二手价格锚点 | CNY 3,600，含 NVLink | 约 CNY 7,000-8,000 | 约 0.5x |

这个项目的核心判断很简单：用约一半 RTX 3090 Ti 二手价格，组出双
22GB RTX 2080 Ti + NVLink，并在 LLM 推理真正关心的物理资源上持平甚至
超过 3090 Ti，再通过 vLLM 运行时优化把这些资源转化成真实 token 产出。

这就是本 fork 的首要价值：把老但仍然很强的 Turing 硅片，通过 Marlin、
FlashQLA/FlashInfer/FA2、TurboQuant/INT8 KV、MTP 和 CUDAGraph 集成，
变成一个严肃可用的 27B/31B 级别推理平台。

## 🧩 核心路线

服务形态：

- 本项目追求的是双 2080 Ti 上的极限单并发性能：一个个人 agent 场景、
  一个足够强的 27B/31B 模型，以及这套硬件能稳定承载的最大实用上下文。
- 它不是多租户 serving 集群。多 agent 使用更适合作为排队式工作区隔离，
  而不是并行长 prefill 吞吐。长上下文并发在调好参数后可以安全排队，
  但在这个 TP=2 profile 下实际会被 runtime scheduler 串行化。

状态：🟢 完整支持；🟡 部分支持；🔴 性能退化；⚪ 不支持。

### Qwen3.6 27B 成熟主线

Qwen 系 27B 是这个 fork 的主要生产路线。它在 Marlin 权重、Native MTP、
FP16/INT8 KV、原生 256K 上下文、YaRN 容量 profile 和图像多模态兼容性上
覆盖最完整。
快速路径：Qwen 使用 FlashQLA-SM70-SM75 处理 Gated DeltaNet /
linear-attention prefill，full-attention prefill 走 FlashInfer / FA2，
head_dim=256 路线完整保留，decode 侧使用 Native MTP + CUDAGraph 策略。

| 功能 | FP16 KV | INT8 KV | TurboQuant KV |
|---|---|---|---|
| Marlin 权重路线 | 🟢 FP8/INT4 | 🟢 FP8/INT4 | 🟢 FP8/INT4 |
| Native MTP3 解码 | 🟢 支持 | 🟢 fast 模式 | 🟡 实验路线 |
| 原生 256K 上下文 | 🟢 文本路线 | 🟢 文本路线 | 🟡 不作为预设 |
| YaRN 512K 扩展 | ⚪ 非目标路线 | 🟢 容量路线 | ⚪ 已验证，不作为预设 |
| No-eager / CUDAGraph | 🟢 支持 | 🟢 支持 | 🟢 graph-safety 已修复 |
| 快速 prefill 路线 | 🟢 FlashInfer / FA2 | 🟢 FlashInfer / INT8 path | 🟡 依赖具体路线 |
| 图像多模态 | 🟢 FP8/INT4 路线 | 🟡 仅 INT4 路线 | 🔴 不晋升 |
| 当前预设状态 | 🟢 推荐 | 🟢 推荐 | 🟡 仅实验 |

Mode 说明：FP16 KV 是 safe 服务路径；INT8 KV 是推荐的 fast 容量 / 性能路线。
TurboQuant KV 保留给特定实验，不作为默认 YaRN 预设。具体 profile 名称、实测
吞吐和容量证据统一维护在 [Profile 导引](profiles/README.zh-CN.md)。

### Gemma4 31B 实验路线

Gemma4 31B 保留为第二路线和实验路线。当前实测 profile 远不如 Qwen 成熟：
这个 runtime 下 native MTP 不支持，FP16/default KV 可以 noMTP 跑通 64K，
压缩 KV 256K 路线不晋升。

| 功能 | FP16 KV | INT8 KV | TurboQuant KV |
|---|---|---|---|
| Marlin 权重路线 | 🟢 GPTQ target | 🟢 GPTQ target | 🟢 GPTQ target |
| Native MTP 解码 | ⚪ 不支持 | ⚪ 不支持 | ⚪ 不支持 |
| 原生 256K 上下文 | ⚪ 未验证通过 | 🔴 不支持 | 🔴 不支持 |
| YaRN 512K 扩展 | ⚪ 不支持 | ⚪ 不支持 | ⚪ 不支持 |
| No-eager / CUDAGraph | 🟢 noMTP 路线 | 🔴 初始化 / fallback 问题 | 🔴 admission 受限 |
| 快速 prefill 路线 | 🟡 慢速 64K 路线 | 🔴 不晋升 | 🔴 不晋升 |
| 图像多模态 | ⚪ 不晋升 | ⚪ 不支持 | ⚪ 不支持 |
| 当前预设状态 | 🟡 仅实验 | 🔴 不晋升 | 🔴 不晋升 |

## 🧪 已测试模型权重

这一节记录 checkpoint 级别的验证结果。这里的标准比“vLLM 能加载”更严格：
支持表示可以启动并生成；推荐表示在双 2080 Ti 上同时具备有意义的速度 /
上下文权衡。

| 模型路线 | 权重路线 | 模型卡 | 状态 |
|---|---|---|---|
| Qwen3.6 27B FP8 | FP8 | [Jackrong/Qwopus3.6-27B-v2-FP8](https://huggingface.co/Jackrong/Qwopus3.6-27B-v2-FP8) | 🟢 推荐 |
| Qwen3.6 27B INT4 | INT4 | [mconcat/Qwopus3.6-27B-v2-AWQ-4bit](https://huggingface.co/mconcat/Qwopus3.6-27B-v2-AWQ-4bit)<br>[QuantTrio/Qwen3.6-27B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ)<br>[llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4) | 🟢 推荐 |
| Qwen3.6 27B AutoRound | AutoRound INT8 | [Minachist/Qwen3.6-27B-INT8-AutoRound W8A16-GS128](https://huggingface.co/Minachist/Qwen3.6-27B-INT8-AutoRound/tree/W8A16-GS128) | 🟡 支持 |
| Gemma4 31B GPTQ | GPTQ-INT4 + assistant draft | [ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ](https://huggingface.co/ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ) | 🟡 支持 |

FP8 是推荐的高质量 Qwen 8bit 路线；INT4 仍然是默认性能 / 容量路线。
AutoRound INT8 是实验 checkpoint 路线。

## 🛠️ 目标硬件与运行环境

- 已验证 GPU profile：双 RTX 2080 Ti 22GB，SM75，NVLink，tensor parallel
  size 2
- CUDA/PyTorch：CUDA 12.8，`torch 2.11.0+cu128`
- Fork 发布版本：`v0.1.5`
- 基础 vLLM：`0.21.0`
- 仓库身份：`vllm-2080ti-definitive`
- 运行时身份：`vllm-sm75-tp2-cu128`
- 兼容目标：NVIDIA Turing / SM75 显卡。其它 Turing 显卡仍需要按显存容量、
  P2P/NVLink 行为、模型 head_dim、KV dtype、CUDAGraph/MTP 设置重新验证
  profile。

## 🚀 如何使用

源码 checkout 后直接运行：

```bash
./build.sh
./launcher.sh
```

然后在 launcher 里选择三件事：

1. checkpoint 目录
2. Profile 路径，先看 [profiles/README.zh-CN.md](profiles/README.zh-CN.md)
3. 端口和仅本地 / 局域网访问

启动成功后会打印 OpenAI-compatible API 地址。非交互启动示例：

```bash
MODEL_DIR=/path/to/qwen-or-gemma-checkpoint \
PROFILE=qwopus36-27b/safe/int4/fp16kv-256K-mtp3-text-only.env \
MODE=safe \
PORT=8000 \
SERVICE_SCOPE=lan \
CUDA_VISIBLE_DEVICES=0,1 \
./launcher.sh --non-interactive
```

Profile 只声明兼容模式，不再提供推荐启动模式。需要指定模式时，显式传
`MODE=safe`、`MODE=normal` 或 `MODE=fast`；launcher 会根据 profile 做二次校验。

### 本机已验证启动配置

下面是这台双 2080 Ti 机器上当前主力、已实测启动成功的路线，忘了怎么启动时
直接照抄即可（`./build.sh` 已跑过、`.venv` 已存在时无需重跑）。

交互式（默认，在终端里直接运行）：

```bash
./launcher.sh
```

进入菜单后依次选：`1` checkpoint 目录 → `2` Profile 路径 → `4` 模式 →
`5` 端口 → `8` 启动；停止选 `9`。

非交互式一条命令（等价于上面这套配置）：

```bash
cd /home/david/work/vLLM-2080Ti-Definitive
MODEL_DIR=/home/david/work/models/Qwopus3.6-27B-v2-FP8 \
PROFILE=qwopus36-27b/user/qwen27b-fp8-fp16kv-64K-mtp3-text-only.env \
MODE=safe PORT=8000 GPU_DEVICES=0,1 TP_SIZE=2 \
bash ./launcher.sh --non-interactive
```

该路线的实际参数（以 profile 与 `--print-config` 输出为准，profile 文件名里的
`64K` 只是命名，并非真实上下文长度）：

| 项 | 值 |
|---|---|
| 权重 | Qwopus3.6-27B-v2-FP8（`QUANTIZATION=fp8`） |
| KV 精度 | FP16/default（质量路线） |
| 上下文 | `MAX_MODEL_LEN=48000` |
| MTP | `MTP_K=3`（safe 模式 + FP16 KV，可用） |
| 并发 | `MAX_NUM_SEQS=1`（极限单并发） |
| GPU / TP | `0,1`，`TP_SIZE=2` |
| 模式 | `safe` |
| 工具调用 | `ENABLE_TOOL_CALLING=1`，`TOOL_CALL_PARSER=qwen3_xml` |
| API | `http://127.0.0.1:8000/v1`（`SERVICE_SCOPE=local`） |

启动过程说明：`launch_server` 会用 `nohup` 后台拉起 api_server，写入
`run-logs/<served-name>.pid`，等待就绪（引擎超时 `VLLM_ENGINE_READY_TIMEOUT_S=1800`）
并跑一次 smoke 测试，全部通过才打印 `START OK`。日志在
`run-logs/vllm-<served-name>-<时间戳>.log`。

验证与停止：

```bash
# 看是否存活 / 已加载模型
curl -s http://127.0.0.1:8000/v1/models
# 停止：交互菜单选 9，或直接 kill pid 文件里的进程
cat run-logs/vllm-qwen27b-fp8-fp16kv-64K-mtp3-text-only-cu128.pid | xargs -r kill
```

> 注意：非交互模式不会自动读取历史 `run-logs/start-manager.state`，`MODEL_DIR`、
> `GPU_DEVICES`、`TP_SIZE`、`MODE`、`PORT` 这些全局项需在命令里显式给出；
> profile 文件本身只携带单条路线的参数。

## 🤖 与 Hermes Agent 配合使用

本仓库负责把模型推理服务跑起来，上层的 agentic 使用（工具调用、多轮编程
任务）由 [Hermes Agent](https://hermes-agent.nousresearch.com)（Nous Research
`hermes-agent`）承担。Hermes 是一个独立的 pip CLI，不是本仓库的一部分，它通过
OpenAI 兼容接口消费本仓库启动的 vLLM。二者是“服务 / 客户端”关系。

- 安装位置：conda 环境 `hermes`，可执行入口 `hermes`（`hermes-agent`）。
- 配置主目录：`~/.hermes/`（`config.yaml` 模型/工具、`SOUL.md` 人格、
  `sessions/`、`memories/`、`skills/` 等）。
- `~/.hermes/bin/tirith` 是配套的 URL 安全网关（执行 shell 前做安全检查），
  不是 agent 本体。

### 启动顺序（两步）

**第一步**：先按[本机已验证启动配置](#本机已验证启动配置)把 vLLM 跑起来，并确认
服务名已加载：

```bash
curl -s http://127.0.0.1:8000/v1/models | grep -o '"id":"[^"]*"'
# 预期输出："id":"qwen27b-fp8-fp16kv-64K-mtp3-text-only-cu128"
```

**第二步**：启动 Hermes。

```bash
conda activate hermes
cd ~/.hermes
hermes                       # 交互式 chat
# 常用变体：
hermes --tui                 # 终端 UI
hermes chat -Q -q "你的问题"  # 一次性非交互（脚本用）
hermes --resume <SESSION>    # 恢复历史会话
hermes status                # 查看各组件状态
```

### 关键配置：服务名必须对齐

Hermes 请求时会把 `config.yaml` 里的 `model.default` 作为模型名发给 vLLM，
该名字必须等于 vLLM 实际加载的 `SERVED_NAME`，否则报 model-not-found。

本机 `~/.hermes/config.yaml` 已对齐为当前 MTP3 路线（旧值以注释保留，便于回退）：

```yaml
model:
  default: qwen27b-fp8-fp16kv-64K-mtp3-text-only-cu128
  provider: custom
  base_url: http://localhost:8000/v1
  api_key: custom
  context_length: 64000
```

- `base_url` 指向本仓库的 vLLM（默认 `:8000`）。
- `context_length: 64000` 是一个“谎报值”：Hermes 有“模型上下文不得低于 64K”的
  门槛检查，而 MTP3 路线真实 `MAX_MODEL_LEN=48000`（48K），所以需该 override
  骗过门槛。实际可用上下文仍是 48K，Hermes 压缩阈值（0.3）保证在限制内安全
  使用。不要把这个 64000 当成真实容量。
- 若换了 profile（例如改用 int4 或其它路线）导致服务名变了，要么启动 hermes 时
  用 `-m <新服务名>` 覆盖，要么改 `config.yaml` 的 `model.default`（并同步
  `~/.hermes/CHANGELOG.md`）。

### 验证链路

```bash
conda activate hermes && cd ~/.hermes
hermes chat -Q -q "不要调用任何工具，直接用中文回复：连接成功"
```

能正常返回即说明 Hermes→vLLM 链路通。注意：本路线是极限单并发
（`MAX_NUM_SEQS=1`），Hermes 与其它客户端会排队，不要并发高频请求。

> 提醒：Hermes 的 `config.yaml` / `CHANGELOG.md` 改动不属于本仓库，但在
> `~/.hermes` 这个 git 仓库里单独维护。

## 🧭 Profile 与推荐路线

从 [Profile 导引](profiles/README.zh-CN.md) 开始选。Profile 按
`profiles/<model>/<mode>/<weight>/<route>.env` 组织，例如
`qwopus36-27b/safe/fp8/fp16kv-128K-mtp3-text-only.env` 和
`qwopus36-27b/fast/int4/int8kv-256K-mtp3-text-only.env`。

KV 精度按这个口径理解：FP16/default KV 是质量路线，INT8 KV 是平衡路线，
TQ4NC 是压缩路线。TQK8V4 目前没有在已验证 profile 中体现出优于 INT8 KV
的容量或速度优势，所以不作为正式路线。

模式名称只保留三档：

- `safe`：默认生产模式。优先保证输出稳定性；FP16/default KV 可以使用 MTP，
  量化 KV 必须关闭 MTP。
- `normal`：中间档，主要用于诊断和手动对比。
- `fast`：高性能模式。量化 KV + MTP 走这个模式，但显存和质量风险更高。

实验 profile 放在每个模型目录下，例如
`profiles/qwopus36-27b/experimental/fp8/`。launcher 默认隐藏实验路线；需要尝试时，
把 Profile directory 指到该 experimental 子目录，或设置
`PROFILE_INCLUDE_EXPERIMENTAL=1`。

后续目标 profile 和实测进度记录在
[Profile 蓝图草案](docs/profile-blueprint.zh-CN.md)。

## 🚀 MTP 与 KV 精度

优先使用项目自带 profile，不建议一开始手动调 MTP 和 KV 参数。每条路线的
MTP 已按当前实测选择了更适合部署的值。KV 先按目标选择：FP16/default KV
追求质量，INT8 KV 追求质量 / 容量平衡，TQ4NC 追求最大压缩。

详细 benchmark 记录见
[MTP 任务敏感性](docs/mtp-task-sensitivity.md) 和
[Qwen3.6 KV 吞吐 Sweep](docs/qwen36-kv-throughput-sweep.zh-CN.md)。

## ❓ 硬件 Q&A

**Q：需要什么样的卡间互联？**

A：推荐 NVLink，但真正的底线是 GPU 之间能开启 PCIe P2P。当前验证系统使用了
NVLink，而且 PCIe 拓扑本身很不理想：一张卡 PCIe 3.0 x1，另一张卡 PCIe 3.0
x4。在 NVLink 承担 GPU-to-GPU 通信时，PCIe 插槽带宽不是主要瓶颈。没有
NVLink 时，不能直接认为极窄 PCIe 带宽也足够，仍然需要确认 P2P 行为并按实际
拓扑 benchmark。

**Q：需要很强的 CPU 或很多内存吗？**

A：不需要。已验证路线可以跑在低端桌面 CPU + 16GB RAM 这类较低规格的平台上。
更强 CPU/更大内存主要帮助 compile cache、下载和本地 build，不是
steady-state token generation 的核心瓶颈。

**Q：哪些 Turing 显卡值得尝试？可以 11GB + 22GB 混搭吗？**

A：完整验证目标是双 RTX 2080 Ti 22GB。其它更推荐高显存 TU102 级别显卡：
TITAN RTX 24GB、Quadro RTX 6000 24GB、Quadro RTX 8000 48GB，最好成对使用并
具备 NVLink 或确认可用的 PCIe P2P。不推荐 11GB + 22GB RTX 2080 Ti 混搭来跑
这些 27B/31B profile，因为 vLLM TP=2 基本会被较小 rank 的显存限制。更小的
Turing 卡可以跑小模型，但不是这个 stack 的主要目标。

**Q：已验证的 CUDA、PyTorch 和驱动版本是什么？**

A：已验证 runtime 是 CUDA 12.8 + `torch 2.11.0+cu128`。请使用支持目标 GPU、
并且兼容该 CUDA runtime 的较新 NVIDIA driver。不要随意混用 build/runtime
假设：PyTorch CUDA 版本、本地 CUDA toolkit、FlashInfer/FlashQLA 构建和启动
profile 应保持一致。

**Q：还有哪些硬件风险需要注意？**

A：散热、供电稳定性，以及给模型文件和 compile cache 留够 SSD 空间。长 prefill
或反复 CUDAGraph/AOT 编译时，降频很容易伪装成软件性能回退。

## 🔗 相关项目

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox)：双
  2080 Ti 模型路线、benchmark 汇总、模型记录和运行建议的配套工具箱。
  本仓库则聚焦于 vLLM 运行时源码、补丁和启动配置本身。

## 🙏 致谢 / 上游项目

本仓库是基于上游 [vLLM](https://github.com/vllm-project/vllm) 的硬件定向
fork，遵循 Apache-2.0 license。仓库保留上游项目结构，并加入面向双
2080 Ti / SM75 路线的本地运行时补丁、启动 profile 和验证记录。

当前 runtime 使用或集成的加速组件包括：

- [vLLM](https://github.com/vllm-project/vllm)：基础推理引擎和 serving
  框架。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：vLLM 使用的
  attention、sampling 和量化 kernel 路线。
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)：上游 FlashQLA
  Gated DeltaNet / Qwen3.5 linear-attention 实现。
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75)：
  面向 SM70/SM75 的适配版本，已验证 Qwen3.6 prefill profile 会用到。
- FlashAttention / FA2、TurboQuant、Marlin、CUTLASS、Triton 以及 vLLM
  相关加速 kernel：这些都是已有开源加速工作，本项目将它们整合、适配并在
  目标硬件上验证。

## 🧱 本机环境与构建备忘

这台机器的构建环境（`.venv` 由 `build.sh` 生成，launcher 运行时调用的就是
`RUNTIME_ROOT/.venv/bin/python`）：

```bash
conda activate vllm2080ti
export CUDA_HOME=/home/david/miniconda3/envs/vllm2080ti
export TORCH_CUDA_ARCH_LIST=7.5
export MAX_JOBS=8
./build.sh
```

启动不再需要手动 export 路线参数：`ENFORCE_EAGER`、`GPU_UTIL`、`MTP_K`、KV
精度、`VLLM_ENGINE_READY_TIMEOUT_S`、工具调用解析器等都由所选 profile 决定，
非交互启动时 profile 会覆盖同名环境变量（旧笔记里的 `ENFORCE_EAGER=1`、
`GPU_UTIL=0.9034` 与当前 profile 的 `ENFORCE_EAGER=0`、`GPU_UTIL=0.88` 不一致，
已作废）。完整启动命令见上文[本机已验证启动配置](#本机已验证启动配置)。

