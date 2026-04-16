from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlParallelBlockCfg,
    InstinctRlEncoderActorCriticCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
    InstinctRlNormalizerCfg,
)


@configclass
class MapAttentionBlockCfg(InstinctRlParallelBlockCfg):
    # map component (e.g. depth image with history)
    component_names = ["depth_image"]
    # proprio components to use as query; default will be all other policy components
    proprio_component_names = None
    # latent dim
    d = 64
    # attention heads
    num_heads = 16
    # conv channels for z-processing: [16, d-3]
    conv_channels = [16, 61]
    kernel_sizes = [5, 5]
    paddings = [2, 2]
    nonlinearity = "ReLU"
    output_size = 64
    takeout_input_components = True
    embed_proprio = True


@configclass
class EncoderConfigs:
    # Use our custom encoder class (module:Class)
    class_name = "instinct_rl.modules.map_attention:MapAttentionEncoder"
    map_attention = MapAttentionBlockCfg()


@configclass
class AttentionPolicyCfg(InstinctRlEncoderActorCriticCfg):
    init_noise_std = 1.0
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"

    encoder_configs = EncoderConfigs()
    critic_encoder_configs = EncoderConfigs()


@configclass
class MoEAttentionPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    init_noise_std = 1.0
    # Use 4 experts by default (matching BASE)
    num_moe_experts = 4
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    encoder_configs = EncoderConfigs()
    critic_encoder_configs = EncoderConfigs()
    moe_gate_hidden_dims = []


@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    # Use the WasabiPPO variant (AMP baseline) for discriminator-based AMP features
    class_name = "WasabiPPO"
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
class NormalizersCfg:
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class G1ParkourAttentionAmpPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: AttentionPolicyCfg = AttentionPolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    log_interval = 10
    experiment_name = "g1_parkour_attention_amp"
    resume = False
    load_run = ""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "_attention_amp"


@configclass
class G1ParkourAttentionAmpMoEPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: MoEAttentionPolicyCfg = MoEAttentionPolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    log_interval = 10
    experiment_name = "g1_parkour_attention_amp_moe"
    resume = False
    load_run = ""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "_attention_amp_moe"
