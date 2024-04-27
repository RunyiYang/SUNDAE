#!/usr/bin/env python
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

# !/usr/bin/env python
"""
Starts viewer in eval mode.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal, Union

import tyro

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.viewer.server.viewer_state import ViewerState
from nerfstudio.scripts.gaussian_splatting.gaussian_splatting_config import GaussianSplattingConfig


@dataclass
class RunViewer(GaussianSplattingConfig):
    def main(self) -> None:
        """Main function."""
        pipeline = self.setup_pipeline()
        config = self.config
        config.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        config.viewer.num_rays_per_chunk = 1

        _start_viewer(config, pipeline, 999999999)

    def save_checkpoint(self, *args, **kwargs):
        """
        Mock method because we pass this instance to viewer_state.update_scene
        """


def _start_viewer(config: TrainerConfig, pipeline: Pipeline, step: int):
    """Starts the viewer

    Args:
        config: Configuration of pipeline to load
        pipeline: Pipeline instance of which to load weights
        step: Step at which the pipeline was saved
    """
    base_dir = config.get_base_dir()
    os.makedirs(base_dir, exist_ok=True)
    viewer_log_path = base_dir / config.viewer.relative_log_filename
    viewer_state = ViewerState(
        config.viewer,
        log_filename=viewer_log_path,
        datapath=pipeline.datamanager.get_datapath(),
        pipeline=pipeline,
    )
    banner_messages = [f"Viewer at: {viewer_state.viewer_url}"]

    # We don't need logging, but writer.GLOBAL_BUFFER needs to be populated
    config.logging.local_writer.enable = False
    writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=banner_messages)

    assert viewer_state and pipeline.datamanager.train_dataset
    viewer_state.init_scene(
        train_dataset=pipeline.datamanager.train_dataset,
        train_state="completed",
        eval_dataset=pipeline.datamanager.eval_dataset,
    )
    if isinstance(viewer_state, ViewerState):
        viewer_state.viser_server.set_training_state("completed")
    viewer_state.update_scene(step=step)
    while True:
        time.sleep(0.01)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RunViewer).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RunViewer)  # noqa
