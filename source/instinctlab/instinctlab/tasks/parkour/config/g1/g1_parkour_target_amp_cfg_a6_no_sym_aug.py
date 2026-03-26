import copy

from isaaclab.utils import configclass

from .g1_parkour_target_amp_cfg import G1ParkourEnvCfg, G1ParkourEnvCfg_PLAY


@configclass
class G1ParkourEnvCfgA6NoSymAug(G1ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # A6 ablation: disable symmetric augmentation in motion reference.
        self.scene.motion_reference = copy.deepcopy(self.scene.motion_reference)
        self.scene.motion_reference.symmetric_augmentation_link_mapping = []
        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        self.scene.motion_reference.symmetric_augmentation_joint_reverse_buf = None


@configclass
class G1ParkourEnvCfgA6NoSymAug_PLAY(G1ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.motion_reference = copy.deepcopy(self.scene.motion_reference)
        self.scene.motion_reference.symmetric_augmentation_link_mapping = []
        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        self.scene.motion_reference.symmetric_augmentation_joint_reverse_buf = None
