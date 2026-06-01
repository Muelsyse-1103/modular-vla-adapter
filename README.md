# Prismatic VLA Adapter

一个参考 `VLA-Adapter / VLA-Adapter-Pro` 思路写的小型 Prismatic 风格框架。它不复制完整官方训练工程，而是把核心过程抽象成清晰的流水线：

```text
任意 VLM / MLLM backbone
  -> ActionQuery 插入 + 图像 token 拼接 + hidden-state 切片
  -> 层选择 + hidden 投影 + Raw token 压缩
  -> Bridge Attention policy
  -> 连续 action chunk + 动作反归一化
```

目标不是“所有模型零成本通用”，而是让接入其他大模型时只需要实现一个 `BackboneAdapter`，其余 ActionQuery、Raw/AQ 特征抽取、Bridge policy、L1 训练目标都保持不变。

## 来自参考实现的设计取舍

参考工程里有几个关键事实：

- `ActionQuery` 是可学习 token，替换动作 placeholder embedding 后进入 VLM 序列。
- 图像 token 插在 BOS 后，LLM 输出所有层 hidden states。
- Policy 读取两类条件：Raw visual prefix tokens 和 ActionQuery-aligned hidden states。
- `VLA-Adapter-Pro` 的 policy 使用 self / AQ+proprio / Raw 三路 Bridge Attention，并输出 `[B, H, action_dim]` 连续动作。
- 原代码默认和 Qwen2.5-0.5B 的 24 层、hidden size 896 绑定较深；本框架把这些都放进配置。

## 目录

```text
prismatic_adapter/
├── pipeline.py              # VLAAdapter 完整流水线
├── adapters/                # 新大模型/视觉塔适配入口
├── conditioning/            # 层选择、hidden 投影、Raw token 压缩
├── action_heads/            # Bridge continuous action head
├── datasets/                # AdapterBatch / SampleAdapter / collator
├── training/                # trainer、optimizer、scheduler、LoRA、logging
├── runtime/                 # inference 和 checkpoint
├── config.py
├── sequence.py              # ActionQuery 替换、视觉 token 拼接、hidden-state 抽取
└── types.py
```

旧路径 `backbones/`、`components/`、`policy/` 仍保留作为实现和兼容层；新代码建议从
`adapters/`、`conditioning/`、`action_heads/`、`pipeline.VLAAdapter` 这些更清晰的路径导入。

## 接入一个新大模型

实现 `BackboneAdapter.forward_with_action_queries()`：

1. 用模型 embedding 层把 `input_ids` 转成 `[B, S, D]`。
2. 调用 `replace_masked_embeddings()`，把 `batch.action_mask` 标出的 placeholder 替换为 `[Q, D]` ActionQuery。
3. 用新模型自己的视觉塔得到 `[B, P, D]` visual tokens。
4. 调用 `build_multimodal_embeddings()` 得到 `BOS -> vision -> text/AQ` 的 fused embeddings 和 shifted action mask。
5. 以 `inputs_embeds` 方式调用语言模型，并返回全部 hidden states。

如果你的模型不是 Prismatic/OpenVLA 结构，通常只需要自定义第 3、5 步。

## 最小用法

```python
from prismatic_adapter import (
    AdapterConfig,
    ConditioningConfig,
    PolicyConfig,
    PrismaticAdapterPolicy,
    SequenceConfig,
)

cfg = AdapterConfig(
    sequence=SequenceConfig(action_query_tokens=64),
    conditioning=ConditioningConfig(
        num_condition_layers=24,
        layer_strategy="uniform",
        raw_token_budget=512,
        projection="linear",
    ),
    policy=PolicyConfig(hidden_size=1024, num_layers=24, action_dim=7, action_horizon=8),
    train_backbone=False,
)

policy = PrismaticAdapterPolicy(
    backbone=my_backbone_adapter,
    config=cfg,
    proprio_dim=8,
)

pred_actions = policy(batch)  # [B, 8, 7]
```

`batch.actions` 应该是已经归一化到训练动作空间的 action chunk。推理时按数据集统计做 unnormalize，这部分故意留给外层任务代码处理。

## Qwen3.5 + 标准 ViT 示例

本仓库提供了一个具体实例：[prismatic_adapter/backbones/qwen_vit.py](prismatic_adapter/backbones/qwen_vit.py)。

默认组合：

```text
Language backbone: pretrained_models/Qwen3.5-2B
Vision backbone:   vit_large_patch14_reg4_dinov2.lvd142m + vit_so400m_patch14_siglip_224
Backbone dim:      Qwen text hidden size 2048
Policy dim:        示例中投影到 1024
```

先缓存视觉模型：

```bash
python scripts/download_vision_backbones.py --cache-dir pretrained_models/vision_cache
```

构造示例 policy：

```bash
python examples/qwen35_vit_policy.py
```

## 设计原则

- token index 必须显式：视觉区间、ActionQuery mask、BOS 区间都由 `SegmentSlices` 保存。
- backbone 只提供条件，不知道 policy 细节。
- policy 只吃 `[B, L, R, D]` Raw 和 `[B, L, Q, D]` AQ 条件，不知道 prompt、tokenizer 或图像处理。
- hidden size、层数、Raw token 数都通过适配组件变成固定 policy 输入。
- 默认冻结 backbone，只训练 ActionQuery、Bridge policy 和可选 proprio projector；如需 LoRA，可在外层把 backbone 包成 PEFT 模型后设 `train_backbone=True`。

更详细的结构地图在 [docs/framework_map.md](docs/framework_map.md)，架构说明在
[docs/architecture.md](docs/architecture.md)。

## Training

训练入口已补齐到 `scripts/train_qwen35_vit.py`。它使用 dataset factory 接入外部
RLDS/LIBERO/CALVIN 数据，支持 LoRA、AMP、梯度累积、验证、checkpoint/resume、
JSONL 与可选 W&B 日志。

```bash
python scripts/train_qwen35_vit.py \
  --dataset-factory my_project.datasets:build_libero_dataset \
  --vision-pretrained \
  --use-lora \
  --output-dir outputs/qwen35_vit_libero_object
```

训练代码说明见 [docs/training.md](docs/training.md)。

## Remote Env Evaluation

评估环境可以作为独立 Python 进程运行，通过 ZMQ 与模型/训练进程交互，避免
LIBERO、robosuite、MuJoCo 等依赖和 Qwen/torch/timm 环境冲突。

```text
prismatic_adapter/  # 模型适配
vla_runtime/        # policy rollout / eval runner
env_process/        # 独立环境进程
```

先跑 fake env 闭环：

```bash
python scripts/serve_fake_env.py --endpoint tcp://127.0.0.1:5555
python scripts/eval_with_remote_env.py --endpoint tcp://127.0.0.1:5555
```

LIBERO 后端已经拆到 `env_process/backends/libero.py`，启动入口为：

```bash
python scripts/serve_libero_env.py \
  --endpoint tcp://127.0.0.1:5555 \
  --task-suite libero_object \
  --resolution 256
```

说明见 [docs/remote_env.md](docs/remote_env.md)。
