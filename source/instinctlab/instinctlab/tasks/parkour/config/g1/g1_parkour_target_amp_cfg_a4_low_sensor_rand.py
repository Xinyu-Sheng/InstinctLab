import copy

from isaaclab.utils import configclass

from .g1_parkour_target_amp_cfg import G1ParkourEnvCfg, G1ParkourEnvCfg_PLAY


@configclass
class G1ParkourEnvCfgA4LowSensorRand(G1ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # A4 ablation: reduce visual randomization and delay to test robustness dependence.
        self.scene.camera = copy.deepcopy(self.scene.camera)
        self.scene.camera.noise_pipeline = {
            "crop_and_resize": self.scene.camera.noise_pipeline["crop_and_resize"],
            "depth_normalization": self.scene.camera.noise_pipeline["depth_normalization"],
        }

        self.observations.policy.depth_image.params["delayed_frame_ranges"] = (0, 0)
        self.observations.critic.depth_image.params["delayed_frame_ranges"] = (0, 0)


@configclass
class G1ParkourEnvCfgA4LowSensorRand_PLAY(G1ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.camera = copy.deepcopy(self.scene.camera)
        self.scene.camera.noise_pipeline = {
            "crop_and_resize": self.scene.camera.noise_pipeline["crop_and_resize"],
            "depth_normalization": self.scene.camera.noise_pipeline["depth_normalization"],
        }

        self.observations.policy.depth_image.params["delayed_frame_ranges"] = (0, 0)
        self.observations.critic.depth_image.params["delayed_frame_ranges"] = (0, 0)
