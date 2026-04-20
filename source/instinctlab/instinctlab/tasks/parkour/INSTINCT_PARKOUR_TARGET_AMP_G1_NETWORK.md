# Instinct-Parkour-Target-Amp-G1-v0 网络结构说明

## 1. 总体架构

`Instinct-Parkour-Target-Amp-G1-v0` 的核心网络结构是：

- 输入：`depth_image` + 机器人 proprioceptive 观测
- 编码器：标准 Conv2D depth encoder
- 策略头：actor MLP
- 价值头：critic MLP
- 训练辅助：`WasabiPPO` AMP 判别器用于训练，不参与推理

> 注意：这个任务与 `Instinct-Parkour-Target-Attention-Amp-G1-v0` 不完全相同。后者使用 `MapAttentionEncoder`，而普通 `Amp-G1` 版本当前是用 `DepthEncoderConv2dCfg`。

---

## 2. 模型组成图

```
[原始观测]
    |
    |-- depth_image ------------------> DepthConv2DEncoder
    |
    |-- proprio inputs --------------------------------------+
                                                              |
                                                      [encoder latent]
                                                              |
                                                     concat / combine
                                                              |
                                                     ┌────────┴────────┐
                                                     |                 |
                                                 actor MLP         critic MLP
                                                     |                 |
                                                 actions           value
```

---

## 3. 编码器详细结构

当前 `Instinct-Parkour-Target-Amp-G1-v0` 的 encoder 采用的是 `DepthEncoderConv2dCfg`：

- 类名：`Conv2dHeadModel`
- 输出维度：`128`
- 输入组件：`depth_image`
- 结构参数：
  - `channels = [4]`
  - `kernel_sizes = [3]`
  - `strides = [1]`
  - `paddings = [1]`
  - `hidden_sizes = [256, 256]`
  - `nonlinearity = "ReLU"`
  - `use_maxpool = True`

### 编码器作用

- 把深度图转换成低维特征表示
- 输出一个固定大小的 encoder latent
- 该 latent 与 proprio 观测一起进入后续网络

---

## 4. Actor-Critic 头结构

这个任务使用 `EncoderActorCritic` 类似结构：

- `encoder` 负责把原始观测转换成嵌入特征
- `ActorCritic` 接收 encoder 输出并进行决策

### MLP 结构

常见配置为：

- `actor hidden dims = [256, 128, 64]`
- `critic hidden dims = [256, 128, 64]`
- 激活函数：`elu`

因此：

- actor：`encoded_obs_dim -> 256 -> 128 -> 64 -> action_dim`
- critic：`encoded_obs_dim -> 256 -> 128 -> 64 -> 1`

对于普通 `Amp-G1` 任务，`encoded_obs_dim` 由 encoder 输出决定，通常是 `128`。

---

## 5. 任务差异说明

`ATTENTION_AMP_G1_NETWORK.md` 中描述的 memory 方案，原本是基于 `MapAttentionEncoder` 的：

- depth map -> token -> cross-attention
- proprio 作为 query
- 输出 `map_encoding(64)` + `proprio_emb(64)`

而 `Instinct-Parkour-Target-Amp-G1-v0` 当前使用的是更传统的 Conv2D depth encoder，因此：

- 原始 attention 版本的 token / cross-attention 细节不能直接照搬
- 但“历史记忆”的核心思想仍然可以迁移

---

## 6. 可能的历史记忆改进

这个 memory 设计可以“概念上移植”到 `Instinct-Parkour-Target-Amp-G1-v0`：

### 方案

- 在 encoder 之外保持一个递归 memory 向量，例如 `(B, 64)`
- 每步用当前 encoder latent 和 proprio 更新它
- 将 memory 拼接到 actor/critic 的最终输入

### 具体形式

- `latent_t = encoder(depth_image_t)`
- `proprio_t = proprio_obs_t`
- `memory_t = GRUCell(concat(latent_t, proprio_t), memory_{t-1})`
- `policy_input = concat(latent_t, proprio_t, memory_t)`

### 好处

- 保留历史空间信息
- 提供跨步语义记忆
- 不需要把多帧 depth 堆成大 tensor
- 适合跑酷任务中“当前视野丢失的障碍/台阶”信息

---

## 7. 推荐改进方向

如果你要给 `Instinct-Parkour-Target-Amp-G1-v0` 加 memory，建议：

- 不改 encoder 内部结构，只在 encoder 输出后增加 memory 模块
- 让 memory 通过 GRU 或类似递归结构更新
- 把 memory 拼接到 actor/critic 输入
- 如果需要更强的约束，可加辅助目标：用 `memory_t` 预测未来一段时间的 encoder latent 或地形特征

---

## 8. 结论

- `ATTENTION_AMP_G1_NETWORK.md` 的 memory 思路可以迁移，但要根据 `Amp-G1` 的实际 encoder 结构做适配。
- 对于 `Instinct-Parkour-Target-Amp-G1-v0`，更适合把 memory 加在 `DepthConv2DEncoder` 的 latent 之后，而不是直接复制 `MapAttentionEncoder` 的 attention token 结构。
- 这样既保留了历史记忆价值，又不会破坏当前任务的 encoder 设计。
