#!/usr/bin/env python3
"""
Exercise 4 test runner using:
  - ./Downloads/ConvWithAsymmetricPadding.onnx (if present)
  - ~/Downloads/ConvWithAsymmetricPadding.onnx
or override with env var:
  ONNX_PATH=/path/to/ConvWithAsymmetricPadding.onnx

This ONNX model uses a standard-domain Conv (NCHW). hls4ml's ONNX Conv parser
expects QONNX channels-last Conv nodes, so we apply a minimal patch for testing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from hls4ml_sofie import extract_modelgraph_config, onnx_to_hls4ml_modelgraph


def _get_layer(cfg: dict, *, name: str) -> dict:
    for layer in cfg.get("layers", []):
        if layer.get("name") == name:
            return layer
    raise KeyError(f"Layer not found: {name}")


def main() -> int:
    override = os.environ.get("ONNX_PATH")
    candidates = [
        Path(override) if override else None,
        Path.cwd() / "Downloads" / "ConvWithAsymmetricPadding.onnx",
        Path.home() / "Downloads" / "ConvWithAsymmetricPadding.onnx",
    ]
    onnx_path = next((p for p in candidates if p is not None and p.exists()), None)
    if onnx_path is None:
        raise FileNotFoundError(
            "Could not find ConvWithAsymmetricPadding.onnx. Tried: "
            + ", ".join(str(p) for p in candidates if p is not None)
        )

    mg = onnx_to_hls4ml_modelgraph(
        str(onnx_path),
        output_dir="hls4ml_conv_asym_pad",
        project_name="conv_asym_pad",
        force_qonnx_channels_last=True,
        disable_flows=True,
    )

    cfg = extract_modelgraph_config(mg, include_weights=True)

    # Sanity checks for this specific ONNX model.
    assert cfg["inputs"][0]["name"] == "x"
    assert cfg["inputs"][0]["shape"] == [7, 5, 1]
    assert cfg["outputs"][0]["shape"] == [4, 2, 1]

    conv = _get_layer(cfg, name="Conv_0")
    attrs = conv["attributes"]
    assert conv["class_name"] == "Conv"
    assert attrs["data_format"] == "channels_last"
    assert attrs["filt_height"] == 3 and attrs["filt_width"] == 3
    assert attrs["stride_height"] == 2 and attrs["stride_width"] == 2
    assert attrs["pad_top"] == 1 and attrs["pad_left"] == 0
    assert attrs["pad_bottom"] == 1 and attrs["pad_right"] == 0
    assert attrs["dilation_height"] == 1 and attrs["dilation_width"] == 1
    assert attrs["group"] == 1

    # Ensure JSON stability (fails if any non-serializable types sneak in).
    json.dumps(cfg)

    print(json.dumps(cfg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
