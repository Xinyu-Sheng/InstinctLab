# Attention-Amp-G1-v0 网络结构说明

## 1. 总体架构

`Instinct-Parkour-Target-Amp-G1-v0` 的网络结构是：

- 输入：`depth_image` + 机器人 proprioceptive 输入
- 编码器：`MapAttentionEncoder`
- 策略头：标准 actor MLP
- 价值头：标准 critic MLP
- 训练辅助：AMP 判别器只在训练阶段使用，不属于部署推理网络

相关实现文件：
- `instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_attention_amp_cfg.py`
- `instinct_rl/instinct_rl/modules/map_attention.py`
- `instinct_rl/instinct_rl/modules/encoder_actor_critic.py`
- `instinct_rl/instinct_rl/modules/actor_critic.py`

---

## 2. 模型组成图

```
[原始观测]
    |
    |-- depth_image -----------------
    |                                \
    |                                 > MapAttentionEncoder
    |                                /           |
    |-- proprio inputs -------------            |
                                             [encoded features]
                                                  |
                                         ┌────────┴────────┐
                                         |                 |
                                     actor MLP         critic MLP
                                         |                 |
                                     actions           value
```

编码器内部流程：

```
depth_image -> conv2d -> flatten -> conv_out (H*W x 61)
coords(xyz) -> concat -> tokens (H*W x 64)
proprio -> linear -> query (1 x 64)
query + tokens -> MultiHeadAttention(num_heads=16, d=64)
    -> map_encoding (64)
    -> proprio_emb (64)
concat -> encoded policy obs
```

---

## 3. 编码器详细结构

### `MapAttentionEncoder`

该编码器由一个 block 构成，配置项如下：

- `class_name = "instinct_rl.modules.map_attention:MapAttentionEncoder"`
- `map_attention.d = 64`
- `map_attention.num_heads = 16`
- `map_attention.conv_channels = [16, 61]`
- `map_attention.kernel_sizes = [5, 5]`
- `map_attention.paddings = [2, 2]`
- `map_attention.nonlinearity = "ReLU"`
- `takeout_input_components = True`
- `embed_proprio = True`

### `MapAttentionBlock` 内部流程

1. 从 `depth_image` 提取最后一帧深度图
   - 支持 `(B, num_frames, H, W)` 或 `(B, H, W)` 输入
2. 通过两层 Conv2D，将深度图编码为空间特征
   - 输入通道：1
   - 输出通道：61（即 `d-3`）
3. reshape 为 token 序列：`(B, L, 61)`，其中 `L = H * W`
4. 构建每个像素点的坐标 `(x, y, z)`：`(B, L, 3)`
5. 拼接得到 tokens：`(B, L, 64)`
6. 对 proprio 输入做线性投影，得到 query 向量：`(B, 64)`
7. cross-attention：
   - query: `(B, 1, 64)`
   - key/value: `(B, L, 64)`
   - 输出: `(B, 1, 64)`
8. 输出：
   - `map_encoding`：64 维
   - `proprio_emb`：64 维

### 编码器输出

`MapAttentionEncoder` 的输出包含：

- `parallel_latent_0_map_attention`：64
- `parallel_latent_0_map_attention_proprio`：64

由于 `takeout_input_components=True`，编码器会从输出中移除原始 `depth_image` 和被编码的 proprio 组件。

---

## 4. Actor-Critic 头结构

该任务使用 `EncoderActorCritic`，也就是 `EncoderActorCriticMixin` 与基础 `ActorCritic` 组合。

- `act(observations)`：先通过 encoder 生成 latent，再通过 actor MLP 输出动作均值
- `evaluate(critic_observations)`：先通过 critic encoder（如果存在），再通过 critic MLP 输出价值

### MLP 结构

配置为：

- actor hidden dims: `[256, 128, 64]`
- critic hidden dims: `[256, 128, 64]`
- activation: `elu`

因此结构为：

```
actor:  input_dim -> 256 -> 128 -> 64 -> action_dim
critic: input_dim -> 256 -> 128 -> 64 -> 1
```

其中 `input_dim` 应为编码器输出维度之和，通常为 `64 + 64 = 128`。

---

## 5. 关键特点与直观理解

### 关键点

- `Attention-Amp-G1-v0` 不是纯 MLP 观测网络，而是引入了“视觉-本体交叉注意力”。
- 深度图经过卷积处理后，与 proprio 信息一起进入 attention 机制。
- 最终输出的 latent 特征被送入标准 actor/critic MLP。

### 直观理解

核心思想：

- 用 depth map 构造“地图 token”
- 用 proprio 作为“查询”
- 通过 cross-attention 让机器人将深度图空间信息对齐到当前自身状态
- 把对齐后的特征作为动作和值的输入

---

## 6. AMP 训练相关说明

尽管任务名包含 AMP，AMP 仅是训练阶段的辅助模块。

- 训练算法使用 `WasabiPPO`
- 包含 AMP 判别器用于奖励和训练目标
- 该判别器不是推理网络的一部分

所以部署时的推理网络仍然是：

`MapAttentionEncoder -> ActorCritic`

---

## 7. 最终结构总结

```
输入:
  - depth_image
  - proprio 观测

编码器:
  MapAttentionEncoder
    MapAttentionBlock
        从depth_image
        -> conv(depth_image)
        -> token + xyz
        -> proprio proj
        -> cross-attention
        -> 64-d map_encoding 
        从proprio
        -> 64-d proprio_embedding
        返回向量map_encoding和proprio_embedding
    输出 128-d：depth_image + proprio

输出编码:
  128-d latent 表示

策略/价值头:
  actor MLP [256,128,64] -> action
  critic MLP [256,128,64] -> value
```

---

## 8. 可能的历史记忆改进

当前 `MapAttentionBlock` 只使用最后一帧深度图，因此历史信息会被丢弃。对于跑酷任务，前几帧里看到的台阶和障碍物往往是当前帧无法直接感知的。

### 方案：加一个递归记忆向量

- 在 encoder 之外维护一个 `memory` 向量，形状例如 `(B, 64)`。
- 每一步将当前 `map_encoding`、`proprio_embedding` 和上一时刻 `memory` 连接起来，作为更新输入：
  - `memory_t = MemoryUpdater(concat(map_encoding, proprio_embedding, memory_{t-1}))`
- 当前策略的 encoder 输出再拼接这个 `memory_t`：
  - `encoded_obs = concat(map_encoding, proprio_embedding, memory_t)`

### 这个改动带来的好处

- `memory` 可以跨步保存历史语义，保留“之前看见过但当前看不到”的信息。
- 只需一个额外向量，不需要把所有历史展开成大规模 token。
- 这种方式更像 LLM 的记忆向量，比直接把多帧堆到通道里更容易提取有用特征。

### 关键实现点

- `MemoryUpdater` 用 `GRUCell`。
- episode reset 时必须清零对应的 `memory` 行。
- 这会把 policy 输入维度从 `128` 扩展到 `128 + D_mem`，若 `D_mem = 64`，则变成 `192`。

### 进一步优化

- 如果需要，可以把 `memory` 也作为一个额外的 global token 参与 attention，但这不是必须。
- 先从“memory 输出拼接到 encoder”开始，保证历史信息被保留，再根据效果决定是否让 memory 直接参与 attention。
