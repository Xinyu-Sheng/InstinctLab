from isaaclab.utils import configclass

from .g1_parkour_target_amp_cfg import G1ParkourEnvCfg, G1ParkourEnvCfg_PLAY


@configclass
class G1ParkourEnvCfgA5NoPenetration(G1ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # A5 ablation: disable terrain penetration shaping.
        self.rewards.rewards.volume_points_penetration.weight = 0.0


@configclass
class G1ParkourEnvCfgA5NoPenetration_PLAY(G1ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.rewards.volume_points_penetration.weight = 0.0
