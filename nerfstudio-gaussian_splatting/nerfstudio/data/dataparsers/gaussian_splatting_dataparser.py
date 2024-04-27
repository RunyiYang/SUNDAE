import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type, Dict
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs


class GaussianSplattingDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: GaussianSplatting)
    """target class to instantiate"""
    data: Path = Path()


class GaussianSplatting(DataParser):
    config: GaussianSplattingDataParserConfig

    def get_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> DataparserOutputs:
        dataparser_outputs = DataparserOutputs(
            image_filenames=[],
            cameras=Cameras(
                fx=800.,
                fy=800.,
                cx=400.,
                cy=400.,
                distortion_params=camera_utils.get_distortion_params(0., 0., 0., 0.),
                height=torch.tensor(800),
                width=torch.tensor(800),
                camera_to_worlds=torch.tensor([]),
                camera_type=CameraType.PERSPECTIVE,
            ),
            metadata={
                "data": self.config.data,
            }
        )
        return dataparser_outputs

