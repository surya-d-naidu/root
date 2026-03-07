#!/usr/bin/env python3
"""
Demo: build a tiny hls4ml ModelGraph in-memory (no TensorFlow required),
extract its configuration, and translate it to a SOFIE RModel.

Run from the ROOT source tree:

  PYTHONPATH=/home/surya/root/build/lib:$PYTHONPATH \\
  LD_LIBRARY_PATH=/home/surya/root/build/lib:$LD_LIBRARY_PATH \\
  /home/surya/root/.venv-root/bin/python tmva/sofie_parsers/python/demo_hls4ml_to_sofie.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from hls4ml.model.graph import ModelGraph

from hls4ml_sofie import SofieBuildOptions, extract_modelgraph_config, modelgraph_to_sofie_rmodel


def main() -> int:
    config = {
        "Backend": "Vivado",
        "IOType": "io_parallel",
        # Disable flows so layer kinds stay close to the user model
        "HLSConfig": {"Flows": [], "Model": {"Precision": "ap_fixed<16,6>", "ReuseFactor": 1}},
    }

    w = np.random.randn(5, 4).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)

    layer_list = [
        {"class_name": "Input", "name": "x1", "input_shape": [1, 2]},
        {"class_name": "Input", "name": "x2", "input_shape": [1, 3]},
        {"class_name": "Concatenate", "name": "cat", "axis": 2, "inputs": ["x1", "x2"]},
        {"class_name": "Dense", "name": "dense", "n_in": 5, "n_out": 4, "weight_data": w, "bias_data": b},
        {"class_name": "Reshape", "name": "reshape", "target_shape": [1, 4]},
        {"class_name": "Activation", "name": "relu", "n_in": 4, "activation": "relu"},
    ]

    model = ModelGraph.from_layer_list(config, layer_list, inputs=["x1", "x2"], outputs=["relu"])

    cfg = extract_modelgraph_config(model, include_weights=True)
    print(json.dumps(cfg, indent=2))

    # BONUS: build SOFIE RModel and generate inference code
    rmodel = modelgraph_to_sofie_rmodel(model, options=SofieBuildOptions(model_name="demo_hls4ml"))
    rmodel.Initialize()
    rmodel.Generate()
    rmodel.OutputGenerated("demo_hls4ml.hxx")
    print("Wrote demo_hls4ml.hxx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

