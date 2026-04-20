# Instinct-Parkour-Target-Amp-G1-v0 网络结构说明

## 1. 总体架构

`Instinct-Parkour-Target-Amp-G1-v0` 的网络结构由以下模块组成：

- 输入：`depth_image` + 机器人 proprioceptive 观测
- 编码器：`DepthConv2DEncoder`，将深度图映射到固定维度 latent
- 递归记忆：可选 GRUCell，用于在时间维度上累计历史语义信息
- actor 头：MLP 输出动作分布参数
- critic 头：MLP 输出状态值估计
- 训练辅助：`WasabiPPO` AMP 判别器，用于训练期的域适应/对抗训练，不参与推理

本网络设计目标是保持 encoder 结构简单、高效，同时通过递归记忆增强跨步时序信息。

---

## 2. 模型组成图

```
[depth_image] -----------------> DepthConv2DEncoder -> latent_t

[proprio_obs] ----------------> proprio vector

                          +------------------------------+
                          | concat(latent_t, proprio_t) |
                          +------------------------------+
                                       |
                                       v
                           [optional memory update]
                                       |
                                combined policy input
                                       |
                           +--------------------------+
                           |                          |
                       actor MLP                  critic MLP
                           |                          |
                        actions                     value
```

当启用历史记忆时，`latent_t` 与 `proprio_t` 会先进入一个 GRUCell 进行状态更新，输出 `memory_t`，再与原始特征拼接给 actor/critic。

---

## 3. DepthConv2DEncoder 详细结构

当前 `Instinct-Parkour-Target-Amp-G1-v0` 的 depth encoder 使用的是 `DepthEncoderConv2dCfg` 对应的 `Conv2dHeadModel`。

- 输入：`depth_image`
- 输出：`encoder_latent`
- 输出维度：`128`

常见网络配置如下：

- `channels = [4]`
- `kernel_sizes = [3]`
- `strides = [1]`
- `paddings = [1]`
- `hidden_sizes = [256, 256]`
- `nonlinearity = ReLU`
- `use_maxpool = True`

### 编码器职责

- 将深度图转换为低维特征表示
- 提供稳定的固定大小 latent 供后续策略网络使用
- 保持计算开销低，适合实时跑酷任务

---

## 4. Proprioceptive 观测处理

Proprio 观测通常包括机器人关节角度、关节速度、基座状态等。

- 直接拼接：最简实现是将 `proprio_obs` 作为原始向量直接拼接到 encoder latent 后
- 可选编码：若需要更强表达，可先通过小型 MLP 将 `proprio_obs` 映射到更紧凑的特征空间，再与 encoder latent 拼接

推荐形态：

- `proprio_t = proprio_obs` 或
- `proprio_t = MLP(proprio_obs)`

最终得到的基础 policy 输入为：

- `policy_base = concat(latent_t, proprio_t)`

---

## 5. Actor-Critic 头结构

策略头和价值头都采用标准 MLP 结构，接收组合特征作为输入。

### 推荐配置

- actor hidden dims = `[256, 128, 64]`
- critic hidden dims = `[256, 128, 64]`
- 激活函数 = `ELU`
- 输出：actor -> 动作维度，critic -> 标量值

### 结构示意

- actor：`input_dim -> 256 -> 128 -> 64 -> action_dim`
- critic：`input_dim -> 256 -> 128 -> 64 -> 1`

其中 `input_dim` = `latent_dim + proprio_dim`，若使用 memory 则为 `latent_dim + proprio_dim + memory_dim`。

---

## 6. 历史记忆模块设计

为了补偿单帧深度图丢失的时序空间信息，建议在 encoder 输出后引入一个轻量递归记忆模块。

### 6.1 记忆模块位置

- 编码器输出 `latent_t`
- proprio 观测 `proprio_t`
- 将两者拼接后输入 GRUCell
- GRU 输出 `memory_t`
- 再将 `memory_t` 与原始特征拼接给 actor/critic

### 6.2 具体实现

使用一个单层 GRUCell：

```
input_t = concat(latent_t, proprio_t)
memory_t = GRUCell(input_t, memory_{t-1})
policy_input = concat(latent_t, proprio_t, memory_t)
```

- `latent_t` 维度：128
- `proprio_t` 维度：与 proprio observation 维度相关，可在 32-128 之间
- `memory_t` 维度：建议 64
- 初始状态：每个 episode 开始时将 `memory_0` 置零

### 6.3 训练与推理

- 在训练过程中，让 memory 随时间累积历史语义特征
- 推理时保持相同结构，使用上一帧的 `memory_t` 作为下一帧的初始状态
- 若训练数据包含多步序列，episode 断点处应重置 `memory_t`

### 6.4 代码示例

伪代码如下：

```python
latent_t = depth_encoder(depth_image_t)  # [B, 128]
proprio_t = proprio_processor(proprio_obs_t)  # [B, P]
memory_input = torch.cat([latent_t, proprio_t], dim=-1)  # [B, 128+P]
memory_t = gru_cell(memory_input, memory_{t-1})  # [B, 64]
policy_input = torch.cat([latent_t, proprio_t, memory_t], dim=-1)
action = actor(policy_input)
value = critic(policy_input)
```

---

## 7. AMP 判别器与训练辅助

`WasabiPPO` AMP 判别器是训练辅助模块，用于增强策略对环境变化的鲁棒性。该模块仅在训练阶段生效：

- 通过判别器提供额外 AMP 信号
- 有助于域适应 / 对抗训练
- 不参与推理路径

在推理时，网络结构只保留 encoder、memory（若启用）、actor 和 critic。

---

## 8. 进一步优化建议

### 8.1 只改 encoder 输出后端

- 不要修改 depth encoder 的核心结构
- 在 encoder 输出后加入 memory 或 proprio 处理模块
- 这样保持原有 depth 表示能力，同时增强时序信息

### 8.2 memory 模块的辅助目标

若希望更强地约束记忆表示，可增加辅助损失：

- 用 `memory_t` 预测下一帧的 encoder latent
- 或用 `memory_t` 预测未来短期地形特征

这类目标可以增强 memory 对未来环境变化的预见能力。

### 8.3 可扩展性

- 若后续希望强化空间理解，可采用小型 attention 或时间卷积，但当前最简可靠方案是 GRUCell 级联
- 若 proprio 维度较大，可先用 `MLP(proprio_obs)` 将其映射到固定维度，再与 `latent_t` 拼接

---

## 9. 总结

`Instinct-Parkour-Target-Amp-G1-v0` 的核心网络为：

- `DepthConv2DEncoder` 提取深度特征
- `proprio_obs` 与 encoder latent 拼接
- actor/critic MLP 输出动作和值估计
- 可选 GRU 记忆模块增强跨步时序信息
- `WasabiPPO` AMP 判别器仅作训练辅助

该设计保持了传统 Conv2D depth encoder 的简单性与实时性，并通过在 encoder 之后增加轻量递归 memory 来改善历史信息建模。
