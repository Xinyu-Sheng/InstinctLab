from __future__ import annotations

import numpy as np
import os
import torch
from typing import TYPE_CHECKING

import onnxruntime as ort

if TYPE_CHECKING:
    from typing import Callable


def load_parkour_onnx_model(
    model_dir: str, get_subobs_func: Callable, depth_shape: tuple, proprio_slice: slice
) -> Callable:
    """Load the ONNX model as policy, but only for parkour task setting."""
    ort_providers = ort.get_available_providers()
    actor = ort.InferenceSession(
        os.path.join(model_dir, "actor.onnx"), providers=ort_providers
    )
    actor_input_name = actor.get_inputs()[0].name

    # Attention-based parkour exports a full-observation encoder instead of a depth-only encoder.
    attention_encoder_path = os.path.join(model_dir, "0-map_attention.onnx")
    if not os.path.exists(attention_encoder_path):
        legacy_attention_encoder_path = os.path.join(
            model_dir, "map_attention_encoder.onnx"
        )
        if os.path.exists(legacy_attention_encoder_path):
            attention_encoder_path = legacy_attention_encoder_path

    if os.path.exists(attention_encoder_path):
        encoder = ort.InferenceSession(attention_encoder_path, providers=ort_providers)
        encoder_input_name = encoder.get_inputs()[0].name

        def policy(obs: torch.Tensor) -> torch.Tensor:
            encoder_input = obs.cpu().numpy().astype(np.float32)
            encoder_output = encoder.run(None, {encoder_input_name: encoder_input})[0]
            actor_output = actor.run(None, {actor_input_name: encoder_output})[0]
            return torch.from_numpy(actor_output).to(obs.device)

        return policy

    encoder = ort.InferenceSession(
        os.path.join(model_dir, "0-depth_encoder.onnx"), providers=ort_providers
    )

    def policy(obs: torch.Tensor) -> torch.Tensor:
        depth_image_input = get_subobs_func(obs)
        depth_image_input = depth_image_input.cpu().numpy()
        depth_image_input = depth_image_input.reshape((-1, *depth_shape))
        depth_image_output = encoder.run(
            None, {encoder.get_inputs()[0].name: depth_image_input}
        )[0]
        actor_input = np.concatenate(
            [
                obs.cpu().numpy()[:, proprio_slice],
                depth_image_output,
            ],
            axis=1,
        )
        actor_output = actor.run(None, {actor_input_name: actor_input})[0]
        return torch.from_numpy(actor_output).to(obs.device)

    return policy
