from typing import Tuple

from dataclasses import dataclass, field, fields

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.data.datamanagers.gaussian_splatting_datamanager import GaussianSplattingDatamanagerConfig
from nerfstudio.data.datasets.gaussian_splatting_dataset import GaussianSplattingDataset
from nerfstudio.pipelines.gaussian_splatting_pipeline import GaussianSplattingPipelineConfig
from nerfstudio.models.gaussian_splatting import GaussianSplattingModelConfig


@dataclass
class GaussianSplattingConfig:
    config: TrainerConfig = TrainerConfig(
        method_name="gaussian_splatting",
        steps_per_eval_batch=999999999,
        steps_per_save=999999999,
        max_num_iterations=999999999,
        mixed_precision=True,
        pipeline=GaussianSplattingPipelineConfig(
            datamanager=GaussianSplattingDatamanagerConfig(),
            model=GaussianSplattingModelConfig(
                eval_num_rays_per_chunk=1,
                
            ),
        ),
        optimizers={},
        viewer=ViewerConfig(
            max_num_display_images=0,  # camera visualization slower the web viewer, even with 'hide scene/images'
        ),
        vis="viewer",
    )

    model_path: str = None

    load_iteration: int = -1

    auto_reorient: bool = True
    "auto reorient the scene"

    ref_orientation: str = None
    "use specific image as the reference orientation"

    def get_pipeline_setup_arguments(self):
        return {
            "model_path": str(self.model_path),
            "load_iteration": self.load_iteration,
            "auto_reorient": self.auto_reorient,
            "ref_orientation": self.ref_orientation,
        }

    def setup_pipeline(self):
        return self.config.pipeline.setup(
            device="cuda",
            **self.get_pipeline_setup_arguments(),
        )
