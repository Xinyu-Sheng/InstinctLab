# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.parkour.config.g1"


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-ZeroDepth-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfgZeroDepth",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)

# ==========================================

gym.register(
    id="Instinct-Parkour-Target-Amp-G1-A1-NoAmp-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg_a1_no_amp:G1ParkourPPORunnerCfgA1NoAmp",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-A2-NoDepth-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg_a2_no_depth:G1ParkourPPORunnerCfgA2NoDepth",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-A3-Moe1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg_a3_moe1:G1ParkourPPORunnerCfgA3Moe1",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-A4-LowSensorRand-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg_a4_low_sensor_rand:G1ParkourEnvCfgA4LowSensorRand",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-A5-NoPenetration-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg_a5_no_penetration:G1ParkourEnvCfgA5NoPenetration",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-A6-NoSymAug-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg_a6_no_sym_aug:G1ParkourEnvCfgA6NoSymAug",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Parkour-Target-Amp-G1-A7-Recurrent-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg_a7_recurrent:G1ParkourPPORunnerCfgA7Recurrent",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Attention-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_attention_cfg:G1ParkourAttentionPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Attention-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_attention_cfg:G1ParkourAttentionPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Attention-Amp-Old-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_attention_amp_cfg:G1ParkourAttentionAmpPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Attention-Amp-Old-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_attention_amp_cfg:G1ParkourAttentionAmpPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Attention-Amp-Old-MoE-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_attention_amp_cfg:G1ParkourAttentionAmpMoEPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Attention-Amp-Old-MoE-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_attention_amp_cfg:G1ParkourAttentionAmpMoEPPORunnerCfg",
    },
)
