# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A pipeline that dynamically chooses the number of rays to sample.
"""
import os
import json

from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import numpy as np
import torch
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.data.datamanagers.gaussian_splatting_datamanager import GaussianSplattingDatamanager, \
    GaussianSplattingDatamanagerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipelineConfig


@dataclass
class GaussianSplattingPipelineConfig(VanillaPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: GaussianSplattingPipeline)


class GaussianSplattingPipeline(Pipeline):
    """Pipeline with logic for changing the number of rays per batch."""

    config: GaussianSplattingPipelineConfig
    datamanager: GaussianSplattingDatamanager
    dynamic_num_rays_per_batch: int

    def __init__(
            self,
            config: VanillaPipelineConfig,
            device: str,
            auto_reorient,
            ref_orientation,
            **kwargs,
    ):
        super().__init__()
        self.model_path = kwargs["model_path"]

        orientation_transform = None
        camera_pose_transform = None
        # get orientation transform matrix
        if ref_orientation is not None:
            orientation_transform = self.get_orientation_transform_from_image(ref_orientation)
            camera_pose_transform = torch.linalg.inv(orientation_transform)
        elif auto_reorient is True:
            camera_pose_transform = self.get_orientation_transform_by_up()
            orientation_transform = torch.linalg.inv(camera_pose_transform)

        kwargs["orientation_transform"] = orientation_transform
        self.datamanager = GaussianSplattingDatamanager(kwargs["model_path"], camera_pose_transform)
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            **kwargs,
        )
        self.model.to(device)

    def get_eval_image_metrics_and_images(self, step: int):
        pass

    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        pass

    def get_param_groups(self):
        return {}

    def get_orientation_transform_from_image(self, ref_orientation: str):
        # load camera information
        cameras_json_path = os.path.join(self.model_path, "cameras.json")
        if os.path.exists(cameras_json_path) is False:
            raise RuntimeError("{} not exists".format(cameras_json_path))
        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)

        # find specific camera by image name
        ref_camera = None
        for i in cameras:
            if i["img_name"] != ref_orientation:
                continue
            ref_camera = i
            break
        if ref_camera is None:
            raise ValueError("camera {} not found".format(ref_orientation))

        def rx(theta):
            return np.matrix([
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1]
            ])

        # get camera rotation
        rotation = np.eye(4)
        rotation[:3, :3] = np.asarray(ref_camera["rotation"])
        rotation[:3, 1:3] *= -1

        transform = np.matmul(rotation, rx(-np.pi / 2))

        return torch.tensor(transform, dtype=torch.float)

    def get_orientation_transform_by_up(self):
        cameras_json_path = os.path.join(self.model_path, "cameras.json")
        if os.path.exists(cameras_json_path) is False:
            raise RuntimeError("{} not exists".format(cameras_json_path))
        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)

        up = np.zeros(3)
        for i in cameras:
            pose = np.eye(4)
            pose[:3, :3] = np.asarray(i["rotation"])
            pose[:3, 3] = np.asarray(i["position"])
            pose[:3, 1:3] *= -1  # flip the y and z axis

            up += pose[0:3, 1]

        up = up / np.linalg.norm(up)
        R = GaussianSplattingPipeline.rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1

        return torch.tensor(R, dtype=torch.float)

    @staticmethod
    def rotmat(a, b):
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        # handle exception for the opposite direction input
        if c < -1 + 1e-10:
            return GaussianSplattingPipeline.rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))
