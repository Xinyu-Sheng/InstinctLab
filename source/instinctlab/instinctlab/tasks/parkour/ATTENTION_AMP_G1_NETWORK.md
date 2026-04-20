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
  - depth_image: (B, num_frames, H, W) 或 (B, H, W)
  - proprio 观测: (B, proprio_dim)

编码器:
  MapAttentionEncoder
    MapAttentionBlock
        从 depth_image 提取最后一帧: z (B, H, W)
        -> conv(depth_image): (B, 61, H, W)
        -> token + xyz: tokens (B, H*W, 64)
        -> proprio proj: proprio_emb (B, 64)
        -> cross-attention
             query: (B, 1, 64)
             key/value: (B, H*W, 64)
        -> 64-d map_encoding: (B, 64)
        从 proprio -> 64-d proprio_embedding: (B, 64)
        返回向量 map_encoding 和 proprio_embedding
    输出 128-d: encoded_obs (B, 128)

输出编码:
  128-d latent 表示: (B, 128)

策略/价值头:
  actor MLP [256,128,64] -> action
    输入: (B, 128)
    输出: (B, action_dim)
  critic MLP [256,128,64] -> value
    输入: (B, 128)
    输出: (B, 1)
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

## 9. 推荐实现：短期记忆与未来 `map_encoding` 预测（0.5–1s）

下面是一个针对当前跑酷任务（G1, control_dt = sim.dt * decimation = 0.005 * 4 = 0.02s）可直接落地、低成本且高收益的实现方案：

- 目标：训练一个小型递归记忆向量 `memory_t (B, D_mem)`，并让它预测未来 0.5–1 秒内的 `map_encoding`（即 `map_encoding_{t+N}`，N≈25–50 步）。通过辅助 MSE loss 引导 `memory` 保存短期对决策有价值的地形信息。

设计要点（概述）：
- 把 `memory` 的管理放在 `EncoderActorCriticMixin` 层（而非 `MapAttentionEncoder`），保持 encoder 无状态、易导出。
- 采用 `GRUCell` 作为 `MemoryUpdater`：输入为 `concat(map_encoding, proprio_emb)`，隐藏态为 `memory_{t-1}`，输出 `memory_t`。
- 将 `memory_t` 与 `map_encoding`、`proprio_emb` 拼接，作为 actor/critic 的最终输入。
- 在 rollout 中保存每步的 `map_encoding` 用于构造 t→t+N 的标签；跨 episode 的样本不用于 aux loss（用 mask 跳过）。

具体实现细节（代码位置与建议）：
- 在 `instinct_rl/instinct_rl/modules/map_attention.py`：确认 `map_encoding` 维度 `map_dim = 64`（无需改动）。
- 在 `instinct_rl/instinct_rl/modules/encoder_actor_critic.py`：
  - 在 `__init__` 中新增属性：
    - `self.memory_dim = D_mem  # 推荐 64`
    - `self.memory_updater = nn.GRUCell(input_size=map_dim + proprio_dim, hidden_size=D_mem)`
    - `self.memory_predictor = nn.Sequential(nn.Linear(D_mem, 64), nn.ReLU(), nn.Linear(64, map_dim))`
  - 在 `act()`（或 `backbone_act`）流程中：
    1. `obs = self.encoders(observations)` 获得 `map_encoding` 与 `proprio_emb`（从 `obs` 的 slice 中取出）。
    2. `memory = self.memory_updater(torch.cat([map_encoding, proprio_emb], dim=-1), memory)`（memory 为每 env 的隐藏态，shape `(num_envs, D_mem)`）。
    3. 把 `memory` 拼回 `obs`：`obs_with_memory = concat(map_encoding, proprio_emb, memory)` 并传给 `super().act(obs_with_memory)`。
  - 在 `reset(dones)` 或 `ActorCritic.reset(dones)` 中把对应 env 的 `memory` 清零或重置。

Rollout / 存储改动：
- 在 `instinct_rl/instinct_rl/storage/rollout_storage.py` 的构造中新增 `self.map_encodings = torch.zeros(T, num_envs, map_dim, device=self.device)`。
- 在采集每步 transition 时，把 `map_encoding` 一并写入 storage（可以在 `PPO.act()` 或 runner 的 `rollout_step()` 中把 encoder 输出拆出并写入 `RolloutStorage.Transition`）。
- 在构造 minibatch 时，为每个样本 t 计算目标索引 t+N；若 t+N 超出 episode 範围或者遇到 done，则用 mask 跳过该样本的 aux loss。

训练与辅助损失（PPO 端）：
- 在 `instinct_rl/instinct_rl/algorithms/ppo.py` 的 `compute_losses()` 中增加：
  - 从 minibatch 读取 `memory_t`（或在 forward 里让 actor_critic 在调用时把 memory 输出为可取字段）；
  - `pred = self.actor_critic.memory_predictor(memory_t)`；
  - `aux_loss = mse(pred, target_map_encoding)`（对有效样本做平均，使用 mask）。
  - 总 loss：`loss_total = surrogate_loss + value_loss_coef * value_loss + lambda_aux * aux_loss`，推荐初始 `lambda_aux = 0.01`（可在 0.005–0.02 区间搜索）。

## 10. 可行性评估与修改范围

这个 memory+future `map_encoding` 预测方案在当前代码架构下是可行的，改动点集中且实现难度适中。

### 为什么可行

- `EncoderActorCriticMixin` 已经负责把 encoder 输出转换成 policy 输入，因此在这里插入 `memory` 最自然。
- `ActorCritic` 直接按照 `obs_format` 计算 MLP 输入维度，不需要改其内部结构。
- `MapAttentionEncoder` 只需保持当前输出结构，`map_encoding` 本身已经是一个适合预测的 distilled 语义特征。
- `RolloutStorage` 结构允许新增字段，已经具备保存额外过渡信息的基础。

### 修改范围

**核心改动**
- `instinct_rl/instinct_rl/modules/encoder_actor_critic.py`
  - 新增 `memory_updater` 和 `memory_predictor`
  - 在 `act()` 中更新 `memory_t`
  - 把 `memory_t` 拼入 encoder 输出并保证 `obs_format` 同步
  - 在 `reset(dones)` 中清零 memory

- `instinct_rl/instinct_rl/storage/rollout_storage.py`
  - 新增 `map_encodings` 或相似字段
  - 记录每步 `map_encoding`
  - 如果要跨 rollout，考虑使用已有的 `QueueRolloutStorage` 机制

- `instinct_rl/instinct_rl/algorithms/ppo.py`
  - 计算 `memory` 预测 loss
  - 加入 `lambda_aux` 权重
  - 处理跨 episode 的 mask

**可选优化**
- `instinct_rl/instinct_rl/modules/map_attention.py`：确认输出组件名称，不做必要性修改
- 配置文件：增加记忆/预测相关超参
- 如果跨 rollout 训练，需要额外管理 episode 级缓存或 `QueueRolloutStorage`

### 主要风险点

- `rollout` 长度必须足够覆盖预测 horizon，否则有效 aux 样本变少。
- `done` 会打断标签连续性，必须正确 mask 掉跨 episode 样本。
- `memory` 输入维度改变后，`ActorCritic` 初始化时的 `obs_format` 要同步更新。

### 结论

这个方案不是“架构违背”，而是“当前架构的自然扩展”。

- 推荐实现路径是把记忆放在 `EncoderActorCriticMixin`，让 `MapAttentionEncoder` 保持 stateless。
- 若目标是跨 rollout 训练，可以利用已有的 `QueueRolloutStorage` 或自行实现 episode 级缓存。
- `QueueRolloutStorage` 已经存在于 `instinct_rl/instinct_rl/storage/rollout_storage.py`，可作为实现跨 rollout 训练的自然起点。
- 从改动量看，这是中等复杂度：需要改 3 个核心模块，但无需重写 PPO 算法或 attention encoder。

如果你愿意，我可以继续给出一个更具体的“补丁清单 + 最小可执行实现”方案。

目标时间窗口与超参建议：
- 控制周期：`control_dt = sim.dt * decimation = 0.005 * 4 = 0.02s`（当前 config）。
- 预测步数 N：0.5s → 25 步，1.0s → 50 步。推荐先用 `N = 25`（稳健、较易收敛），再尝试 `N = 50` 做对比实验。
- 平滑窗口 `K`: 推荐从 `K=3` 开始（小窗口平均目标更稳定），也可以在后续实验中对比 `K=1`。
- `D_mem`: 32 或 64，首选 64。
- 预测头隐藏维：64。
- `lambda_aux`: 初始 0.01（如训练不稳定降到 0.005）。
- 梯度裁剪：0.5；目标标准化：对 `map_encoding` 使用 running mean/std 进行归一化后计算 MSE。

稳定性与调试要点：
- 只对同一 episode 内的 t→t+N 对计算 aux loss，避免跨 episode 泄漏；当 `done` 在 t..t+N 之间出现时跳过该样本。
- 观察 aux_loss 曲线与 memory_t 的范数，检查是否退化为常数向量；若退化，降低 `lambda_aux` 或缩短 N。
- 若 aux_loss 能下降但策略收益未提升，考虑把目标改为更直接的前方局部 occupancy/最低高度二值预测（still small dim, 更贴近 foot placement）。

小结与优先级
- 这是一个低改动、低显存开销且与当前 encoder/policy 接口兼容的方案，适合先行实验；在 G1 跑酷任务上优先级高。
- 下一步（可选）：我可以直接为上述文件生成逐-file 的代码补丁并运行一次 smoke-test（单环境前向与 reset 检查）。如果需要，请回复“生成补丁并测试”。
