from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class DepthMemoryAttentionEncoderCfg(InstinctRlConv2dHeadCfg):
    output_size = 128
    channels = [4]
    kernel_sizes = [3]
    strides = [1]
    hidden_sizes = [256, 256]
    paddings = [1]
    nonlinearity = "ReLU"
    use_maxpool = True
    component_names = [
        "depth_image",
    ]
    memory_hidden_size = 128
    memory_state_proj_size = 128
    attention_embed_dim = 8
    attention_num_heads = 2


@configclass
class EncoderConfigs:
    class_name = "instinct_rl.modules.memory_attention_encoder:MemoryAttentionEncoder"
    depth_encoder = DepthMemoryAttentionEncoderCfg()


@configclass
class MoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    class_name = "instinct_rl.modules.memory_attention_actor_critic:MemoryAttentionEncoderMoEActorCritic"
    init_noise_std = 1.0
    num_moe_experts = 4
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    encoder_configs = EncoderConfigs()
    critic_encoder_configs = EncoderConfigs()


@configclass
class AmpAlgoCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "WasabiMemoryAttentionPPO"  # 比普通PPO有额外的 AMP reward，用 actor_state 和 reference_state 训练 discriminator
    discriminator_kwargs = {
        "hidden_sizes": [1024, 512],
        "nonlinearity": "ReLU",
    }

    discriminator_reward_coef = 0.25
    discriminator_reward_type = "quad"
    discriminator_loss_func = "MSELoss"
    discriminator_gradient_penalty_coef = 5.0
    discriminator_optimizer_class_name = "AdamW"
    discriminator_weight_decay_coef = 3e-4
    discriminator_logit_weight_decay_coef = 0.04
    discriminator_optimizer_kwargs = {
        "lr": 1.0e-4,
        "betas": [0.9, 0.999],
    }
    memory_consistency_mse_coef = 0.1
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.006
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class G1ParkourPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    experiment_name = "g1_parkour_memory_attention_amp"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = MoEPolicyCfg()  # 策略网络的结构配置
    algorithm = AmpAlgoCfg()  # 训练算法的配置
