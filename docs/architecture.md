# Architecture

## 1. 数据流

```text
AdapterBatch
  input_ids / attention_mask / action_mask / pixel_values / proprio
        |
        v
BackboneAdapter
  embed input ids
  replace action placeholders with ActionQuery
  encode images into visual tokens
  build fused sequence: BOS -> visual tokens -> prompt + AQ
  run language model with output_hidden_states=True
        |
        v
HiddenStateExtractor
  Raw branch: fused hidden states over visual token slice
  AQ branch: fused hidden states over shifted action mask
        |
        v
Conditioning adapters
  LayerSelector: arbitrary backbone depth -> policy depth
  ConditionProjector: backbone hidden size -> adapter hidden size
  TokenCompressor: arbitrary visual token count -> fixed Raw budget
        |
        v
BridgeActionHead
  action-time latent tokens attend to:
    self tokens
    ActionQuery states + proprio token
    Raw visual tokens
        |
        v
normalized action chunk [B, H, A]
```

## 2. 为什么这样拆

官方 `full-vla-adapter` 参考实现把 ActionQuery、视觉 token 插入、HF forward、hidden-state 切片、policy 输入组织放在同一套 OpenVLA 类和训练脚本里。这对于复现单一配置很直接，但换大模型时会遇到三类耦合：

- 语言模型层数和 policy block 数硬绑定。
- `hidden_states` 的 token 位置靠手写切片推导。
- proprio、ActionQuery、Raw visual tokens 的职责在训练脚本和模型类之间交叉。

本框架把它们拆成稳定接口。新 backbone 只要能提供 fused hidden states 和 segment mask，Bridge policy 就可以复用。

## 3. BackboneAdapter 合约

`BackboneAdapter` 必须返回：

- `hidden_states`: tuple/list of `[B, S_fused, D]`。
- `segments.vision`: fused sequence 中 Raw visual token 区间。
- `segments.action_mask`: fused sequence 中 ActionQuery token 的 bool mask。
- `fused_attention_mask`: 调用语言模型时使用的 attention mask。

推荐所有新模型都尽量支持 `inputs_embeds` 前向。这样 ActionQuery 可以直接替换 embedding，而不需要污染 tokenizer vocabulary。

## 4. 其他部件的接口化

| 部件 | 代码 | 解决的问题 |
| --- | --- | --- |
| `LayerSelector` | `components/conditioning.py` | 不同模型层数不同，统一采样到固定条件层数。 |
| `ConditionProjector` | `components/conditioning.py` | 不同 hidden size 投影到 policy hidden size。 |
| `MeanPoolTokenCompressor` | `components/conditioning.py` | 不同视觉 token 数压缩到固定 Raw token budget。 |
| `ActionNormalizer` | `components/actions.py` | 不同数据集动作尺度、关节维度归一化和反归一化。 |
| `PromptAdapter` | `components/prompts.py` | 不同 tokenizer/chat template 下构造 placeholder 与 `action_mask`。 |
| `SampleAdapter` | `data.py` | RLDS/LIBERO/CALVIN/真实机器人样本统一变成 `AdapterBatch`。 |
| `PaddedBatchCollator` | `data.py` | variable-length prompt pad，同时堆叠图像、动作和 proprio。 |
| `ActionPredictor` | `inference.py` | 推理时输出 normalized action 和环境尺度 action。 |
| checkpoint helpers | `checkpoint.py` | adapter/action head/proprio/backbone 权重分组保存。 |

## 5. 换大模型时主要看什么

- `hidden_size`: 必须和 `PolicyConfig.hidden_size` 一致。
- hidden state tuple 是否包含 embedding state：如果包含，默认 `hidden_state_source="transformer_layers"` 会跳过第 0 个状态。
- 视觉 token 输出是否已经投影到语言 hidden size：如果不是，在 adapter 内加 projector。
- prompt 中 action placeholder 个数必须等于 `SequenceConfig.action_query_tokens`。
- 如果模型无法以 `inputs_embeds` 调用，需要在 adapter 中实现等价的 embedding injection 入口。

## 6. 与 VLA-Adapter-Pro 的关系

这里保留了 VLA-Adapter-Pro 的核心结构：

- 可学习 ActionQuery。
- all-layer Raw/AQ 条件。
- AQ + proprio 作为动作相关 branch。
- Raw branch 使用零初始化 gate。
- Bridge policy 一次输出 action chunk。

同时做了几个工程化改动：

- Policy 层数、hidden size、head 数动态配置。
- token segment 显式保存，避免 magic slice。
- action latent 使用可学习时间 query，初始化很小，替代参考实现里 forward 内临时构造的零 latent。
- 训练 step 与数据集、rollout 环境解耦。
