"""
Dataset.
"""
from __future__ import annotations

import json
import os.path
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from nerfstudio.data.datasets.base_dataset import InputDataset


class GaussianSplattingDataset(InputDataset):

    def __init__(self, model_path: str, orientation_transform):
        Dataset().__init__()

        if orientation_transform is not None:
            orientation_transform = orientation_transform.cpu().numpy()

        # load camera data
        with open(os.path.join(model_path, "cameras.json"), "r") as f:
            camera_data = json.load(f)

        image_key_by_shape = {}
        image_shapes = []
        filenames = []

        fx_list = []
        fy_list = []
        cx_list = []
        cy_list = []
        height_list = []
        width_list = []
        c2w_list = []

        for i in camera_data:
            shape = (i["height"], i["width"])

            # if shape not in image_key_by_shape:
            #     image_key_by_shape[shape] = torch.ones(i["height"], i["width"], 3, dtype=torch.float)

            image_shapes.append(shape)

            fx_list.append(i["fx"])
            fy_list.append(i["fy"])
            cx_list.append(i["width"] / 2)
            cy_list.append(i["height"] / 2)
            height_list.append(i["height"])
            width_list.append(i["width"])

            c2w = np.eye(4)
            c2w[:3, :3] = np.asarray(i["rotation"])
            c2w[:3, 3] = np.asarray(i["position"])
            c2w[:3, 1:3] *= -1

            if orientation_transform is not None:
                c2w = np.matmul(orientation_transform, c2w)

            c2w_list.append(c2w[:3, ...])

            filenames.append(i["img_name"])

        self.metadata = {}

        aabb_scale = 1.
        self.scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        # self.image_key_by_shape = image_key_by_shape
        self.image_shapes = image_shapes
        self.filenames = filenames
        self.cameras = Cameras(
            fx=torch.tensor(fx_list, dtype=torch.float),
            fy=torch.tensor(fy_list, dtype=torch.float),
            cx=torch.tensor(cx_list, dtype=torch.float),
            cy=torch.tensor(cy_list, dtype=torch.float),
            distortion_params=None,
            height=torch.tensor(height_list, dtype=torch.int),
            width=torch.tensor(width_list, dtype=torch.int),
            camera_to_worlds=torch.from_numpy(np.stack(c2w_list, axis=0)),
            camera_type=CameraType.PERSPECTIVE,
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        return np.ones(list(self.image_shapes[image_idx]) + [3], dtype=np.uint8) * 255

    def get_image(self, image_idx: int):
        return torch.ones(list(self.image_shapes[image_idx]) + [3], dtype=torch.float)

    def get_data(self, image_idx: int) -> Dict:
        return {
            "image_idx": image_idx,
            "image": self.get_image(image_idx),
        }

    def get_metadata(self, data: Dict) -> Dict:
        return {}

    @property
    def image_filenames(self) -> List[Path]:
        return self.filenames
