"""
Utilities to inspect an in-memory hls4ml `ModelGraph` and (optionally) translate
simple graphs into a SOFIE `RModel` via PyROOT.

This file is intentionally self-contained:
- `extract_modelgraph_config()` solves Exercise 4.
- `modelgraph_to_sofie_rmodel()` is a minimal Exercise 5 (BONUS) scaffold.

The code uses duck-typing so it can be imported even when `hls4ml` (or ROOT) is
not installed; only the functions that need them will raise at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


Json = Union[None, bool, int, float, str, List["Json"], Dict[str, "Json"]]


def _is_numpy_array(x: Any) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "tobytes")


def _jsonable(x: Any) -> Json:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    # numpy scalar -> python scalar
    if hasattr(x, "item") and callable(x.item):
        try:
            v = x.item()
            if v is None or isinstance(v, (bool, int, float, str)):
                return v
        except Exception:
            pass
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        out: Dict[str, Json] = {}
        for k, v in x.items():
            if isinstance(k, str):
                out[k] = _jsonable(v)
            else:
                out[str(k)] = _jsonable(v)
        return out
    if _is_numpy_array(x):
        return {
            "shape": [int(d) for d in x.shape],
            "dtype": str(x.dtype),
        }
    # Common hls4ml types: NamedType / PrecisionType, Variables, WeightVariables
    for attr in ("name", "class_name"):
        if hasattr(x, attr) and isinstance(getattr(x, attr), str):
            return str(getattr(x, attr))
    return str(x)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _try_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _try_call(fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _weight_stats(arr: Any) -> Dict[str, Json]:
    if not _is_numpy_array(arr):
        return {}
    # Avoid importing numpy; rely on ndarray methods.
    try:
        b = arr.tobytes(order="C")
    except Exception:
        b = b""
    stats: Dict[str, Json] = {
        "shape": [int(d) for d in arr.shape],
        "dtype": str(arr.dtype),
        "sha256": _sha256_bytes(b) if b else None,
    }
    # min/max might fail for non-numeric dtypes
    stats["min"] = _jsonable(_try_call(arr.min, default=None))
    stats["max"] = _jsonable(_try_call(arr.max, default=None))
    return stats


def extract_modelgraph_config(
    modelgraph: Any,
    *,
    include_weights: bool = False,
    include_unquantized_weights: bool = False,
) -> Dict[str, Json]:
    """
    Exercise 4:
    Take an in-memory hls4ml `ModelGraph` and return a JSON-serializable
    configuration that summarizes:
    - backend + IO type + flows
    - model inputs/outputs (names, shapes, type names)
    - a per-layer view (names, kind, connectivity, key attributes)
    - optionally weights metadata (shape/dtype/hash/min/max)

    Notes:
    - This intentionally avoids dumping full weight values unless you extend it.
    - The returned dict is designed to be stable and easy to diff.
    """

    # High-level config
    cfg_obj = _try_getattr(modelgraph, "config")
    cfg_dict = _try_getattr(cfg_obj, "config", {}) if cfg_obj is not None else {}
    backend_obj = _try_getattr(cfg_obj, "backend")
    backend_name = _try_getattr(backend_obj, "name", None) or _try_getattr(backend_obj, "__class__", type("X", (), {})).__name__

    io_type = None
    if cfg_obj is not None:
        io_type = _try_call(cfg_obj.get_config_value, "IOType", default=None)
    if io_type is None and isinstance(cfg_dict, Mapping):
        io_type = cfg_dict.get("IOType")

    flows = _try_getattr(cfg_obj, "flows", None)
    if flows is not None:
        flows = [str(f) for f in flows]

    # I/O variables
    inputs = []
    for v in _try_call(modelgraph.get_input_variables, default=[]) or []:
        inputs.append(
            {
                "name": _try_getattr(v, "name"),
                "shape": _jsonable(_try_getattr(v, "shape")),
                "type": _jsonable(_try_getattr(_try_getattr(v, "type"), "name")),
            }
        )

    outputs = []
    for v in _try_call(modelgraph.get_output_variables, default=[]) or []:
        outputs.append(
            {
                "name": _try_getattr(v, "name"),
                "shape": _jsonable(_try_getattr(v, "shape")),
                "type": _jsonable(_try_getattr(_try_getattr(v, "type"), "name")),
            }
        )

    # Layers
    layers: List[Dict[str, Json]] = []
    for layer in _try_call(modelgraph.get_layers, default=[]) or []:
        # Connectivity in terms of node names
        node_inputs = list(_try_getattr(layer, "inputs", []) or [])
        node_outputs = list(_try_getattr(layer, "outputs", []) or [])

        # Resolve to variable names where possible (useful when flows rename outputs)
        input_vars: List[str] = []
        for i in node_inputs:
            v = _try_call(layer.get_input_variable, i, default=None)
            input_vars.append(_try_getattr(v, "name", i))

        output_vars: List[str] = []
        for o in node_outputs:
            v = _try_call(layer.get_output_variable, o, default=None)
            output_vars.append(_try_getattr(v, "name", o))
        if not output_vars:
            v = _try_call(layer.get_output_variable, default=None)
            if v is not None:
                output_vars = [_try_getattr(v, "name")]

        # Keep only simple attributes (skip big structures / codegen objects)
        attrs_obj = _try_getattr(layer, "attributes", {}) or {}
        attrs: Dict[str, Json] = {}
        if isinstance(attrs_obj, Mapping):
            for k, v in attrs_obj.items():
                if not isinstance(k, str):
                    continue
                if k in {"weight", "bias", "weight_t", "bias_t", "config_cpp", "function_cpp", "include_header"}:
                    continue
                # Prefer compact view
                attrs[k] = _jsonable(v)

        layer_entry: Dict[str, Json] = {
            "name": _try_getattr(layer, "name"),
            "class_name": _try_getattr(layer, "class_name"),
            "inputs": [str(x) for x in node_inputs],
            "outputs": [str(x) for x in node_outputs],
            "input_vars": input_vars,
            "output_vars": output_vars,
            "attributes": attrs,
        }

        if include_weights and isinstance(attrs_obj, Mapping):
            w = attrs_obj.get("weight")
            b = attrs_obj.get("bias")
            # WeightVariable/StaticWeightVariable usually exposes `.data` and `.data_unquantized`
            if w is not None and hasattr(w, "data"):
                layer_entry["weight"] = _weight_stats(w.data)
                if include_unquantized_weights and hasattr(w, "data_unquantized"):
                    layer_entry["weight_unquantized"] = _weight_stats(w.data_unquantized)
            if b is not None and hasattr(b, "data"):
                layer_entry["bias"] = _weight_stats(b.data)
                if include_unquantized_weights and hasattr(b, "data_unquantized"):
                    layer_entry["bias_unquantized"] = _weight_stats(b.data_unquantized)

        layers.append(layer_entry)

    hls4ml_version = None
    try:
        import hls4ml  # type: ignore

        hls4ml_version = getattr(hls4ml, "__version__", None)
    except Exception:
        hls4ml_version = None

    out: Dict[str, Json] = {
        "hls4ml_version": hls4ml_version,
        "backend": str(backend_name) if backend_name is not None else None,
        "io_type": _jsonable(io_type),
        "flows": _jsonable(flows),
        "inputs": inputs,
        "outputs": outputs,
        "layers": layers,
        # Pass through useful shape summaries when present
        "input_shapes": _jsonable(cfg_dict.get("InputShapes") if isinstance(cfg_dict, Mapping) else None),
        "output_shapes": _jsonable(cfg_dict.get("OutputShapes") if isinstance(cfg_dict, Mapping) else None),
    }
    return out


def keras_to_hls4ml_modelgraph(
    keras_model: Any,
    *,
    output_dir: str = "hls4ml_prj",
    project_name: str = "myproject",
    backend: str = "Vivado",
    hls_config: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    Convenience helper for the Exercise 4 workflow:
      Keras model -> hls4ml ModelGraph (in-memory)

    The recommended way to obtain `hls_config` is:
      `hls4ml.utils.config_from_keras_model(model, granularity="name")`

    This function is optional and only used if `hls4ml` is installed.
    """
    try:
        import hls4ml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("hls4ml is required for keras_to_hls4ml_modelgraph()") from e

    if hls_config is None:
        hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity="name")

    return hls4ml.converters.convert_from_keras_model(
        keras_model,
        output_dir=output_dir,
        project_name=project_name,
        backend=backend,
        hls_config=dict(hls_config),
        **kwargs,
    )


def _onnx_patch_conv_to_qonnx_channels_last_inplace(onnx_model: Any) -> Any:
    """
    Minimal patch to satisfy hls4ml's ONNX Conv parser, which expects Conv nodes
    to have `node.domain == 'qonnx.custom_op.channels_last'` and channels-last
    shapes.

    This is NOT a general-purpose layout conversion; it is only intended to make
    small test models parseable for Exercise 4 configuration extraction.
    """
    # Patch Conv node domains (+ ensure missing attrs that trigger hls4ml bugs)
    graph = onnx_model.graph
    for idx, node in enumerate(graph.node):
        if not getattr(node, "name", ""):
            try:
                node.name = f"{getattr(node, 'op_type', 'Node')}_{idx}"
            except Exception:
                pass
        if getattr(node, "op_type", None) == "Conv":
            node.domain = "qonnx.custom_op.channels_last"
            # ONNX default: group = 1
            has_group = any(getattr(a, "name", None) == "group" for a in getattr(node, "attribute", []))
            if not has_group:
                try:
                    from onnx import helper  # type: ignore

                    node.attribute.append(helper.make_attribute("group", 1))
                except Exception:
                    pass
            # hls4ml 1.2.0 has a bug in the Conv parser when `dilations` is missing;
            # avoid it by explicitly setting dilations to the ONNX default [1, 1] (or [1] for 1D).
            has_dilations = any(getattr(a, "name", None) == "dilations" for a in getattr(node, "attribute", []))
            if not has_dilations:
                try:
                    from onnx import helper  # type: ignore

                    # Infer number of spatial dims from kernel_shape if present, else default to 2.
                    kshape = None
                    for a in node.attribute:
                        if a.name == "kernel_shape" and getattr(a, "ints", None):
                            kshape = list(a.ints)
                            break
                    ndim = len(kshape) if kshape is not None else 2
                    node.attribute.append(helper.make_attribute("dilations", [1] * ndim))
                except Exception:
                    # If we cannot patch, let downstream fail loudly.
                    pass

    # Patch graph input/output tensor shapes from NCHW -> NHWC when rank==4
    def _swap_nchw_to_nhwc(value_info) -> None:
        try:
            dims = value_info.type.tensor_type.shape.dim
        except Exception:
            return
        if len(dims) != 4:
            return
        n = dims[0].dim_value
        c = dims[1].dim_value
        h = dims[2].dim_value
        w = dims[3].dim_value
        dims[1].dim_value = h
        dims[2].dim_value = w
        dims[3].dim_value = c
        # keep batch as-is
        dims[0].dim_value = n

    # Only patch non-initializer inputs (data tensors), but it's harmless to patch all.
    for vi in list(getattr(graph, "input", [])) + list(getattr(graph, "output", [])):
        _swap_nchw_to_nhwc(vi)

    return onnx_model


def onnx_to_hls4ml_modelgraph(
    onnx_model_or_path: Any,
    *,
    output_dir: str = "hls4ml_onnx_prj",
    project_name: str = "myproject",
    backend: str = "Vivado",
    io_type: str = "io_parallel",
    hls_config: Optional[Mapping[str, Any]] = None,
    disable_flows: bool = True,
    force_qonnx_channels_last: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Convenience helper for the Exercise 4 workflow:
      ONNX model (.onnx) -> hls4ml ModelGraph (in-memory)

    `force_qonnx_channels_last=True` applies a minimal patch that allows hls4ml
    to parse `Conv` nodes in simple NCHW models (like `ConvWithAsymmetricPadding.onnx`).
    """
    try:
        import hls4ml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("hls4ml is required for onnx_to_hls4ml_modelgraph()") from e

    try:
        import onnx  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("onnx is required for onnx_to_hls4ml_modelgraph()") from e

    model = onnx_model_or_path
    if isinstance(onnx_model_or_path, (str, bytes)):
        model = onnx.load(onnx_model_or_path)

    if force_qonnx_channels_last:
        model = _onnx_patch_conv_to_qonnx_channels_last_inplace(model)

    if hls_config is None:
        # hls4ml expects at least Precision and ReuseFactor; config builders may fail on some models
        hls_config = {"Model": {"Precision": "ap_fixed<16,6>", "ReuseFactor": 1}}
    hls_config = dict(hls_config)

    if disable_flows:
        hls_config["Flows"] = []

    # Ensure required keys exist
    hls_config.setdefault("Model", {})
    hls_config["Model"].setdefault("Precision", "ap_fixed<16,6>")
    hls_config["Model"].setdefault("ReuseFactor", 1)

    return hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir=output_dir,
        project_name=project_name,
        backend=backend,
        io_type=io_type,
        hls_config=hls_config,
        **kwargs,
    )


@dataclass(frozen=True)
class SofieBuildOptions:
    model_name: str = "HLS4ML_Model"
    parsed_time: Optional[str] = None
    tensor_dtype: str = "float"  # currently only float is supported by operators below
    strict: bool = True  # if False, skip unsupported layers instead of raising


def modelgraph_to_sofie_rmodel(modelgraph: Any, *, options: SofieBuildOptions = SofieBuildOptions()):
    """
    Exercise 5 (BONUS):
    Convert a subset of hls4ml `ModelGraph` layers into a SOFIE `RModel` using
    PyROOT. Supported layer/operator mapping (best-effort):
      - Activation(relu) -> ROperator_Relu<float>
      - Activation(elu)  -> ROperator_Elu<float>
      - Dense / PointwiseConv1D -> ROperator_Gemm<float> (with weights+bias)
      - Reshape -> ROperator_Reshape (with constant shape tensor)
      - Concatenate -> ROperator_Concat

    Requirements:
    - `ROOT` importable (PyROOT)
    - ROOT built with `tmva-sofie=ON` and `tmva-pymva=ON` (for Python embedding)
    """

    try:
        import ROOT  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyROOT is required for modelgraph_to_sofie_rmodel()") from e

    # Ensure SOFIE operators are declared
    ROOT.gInterpreter.Declare('#include "TMVA/OperatorList.hxx"')
    S = ROOT.TMVA.Experimental.SOFIE

    parsed_time = options.parsed_time or time.asctime()
    rmodel = S.RModel(options.model_name, parsed_time)

    # Helper: size_t vectors (PyROOT tends to map size_t to unsigned long)
    def _szvec(shape: Sequence[int]):
        return ROOT.std.vector("unsigned long")([int(x) for x in shape])

    # Helper: add float tensor data
    def _add_f32_tensor(name: str, shape: Sequence[int], data_flat: Sequence[float]):
        vec = ROOT.std.vector("float")(list(map(float, data_flat)))
        rmodel.AddInitializedTensor(name, _szvec(shape), vec.data())

    # Helper: add int64 tensor data (int64_t is `long` on this platform)
    def _add_i64_tensor(name: str, shape: Sequence[int], data_flat: Sequence[int]):
        vec = ROOT.std.vector("long")(list(map(int, data_flat)))
        rmodel.AddInitializedTensor(name, _szvec(shape), vec.data())

    # Add inputs
    input_var_names: List[str] = []
    for v in modelgraph.get_input_variables():
        name = str(v.name)
        shape = [int(d) for d in v.shape]
        rmodel.AddInputTensorInfo(name, S.ETensorType.FLOAT, _szvec(shape))
        rmodel.AddInputTensorName(name)
        input_var_names.append(name)

    # Map from layer name -> output variable name(s)
    layer_out_var: Dict[str, str] = {}

    def _primary_output_var(layer) -> str:
        # Prefer variable name; fall back to node output name
        out_name = None
        outs = list(getattr(layer, "outputs", []) or [])
        if outs:
            v = _try_call(layer.get_output_variable, outs[0], default=None)
            out_name = _try_getattr(v, "name", outs[0])
        if out_name is None:
            v = _try_call(layer.get_output_variable, default=None)
            out_name = _try_getattr(v, "name", getattr(layer, "name", "out"))
        return str(out_name)

    # Build operators following `get_layers()` order (already topological for typical models)
    for layer in modelgraph.get_layers():
        layer_name = str(layer.name)
        kind = str(layer.class_name)

        if kind in {"Input", "InputLayer"}:
            layer_out_var[layer_name] = _primary_output_var(layer)
            continue

        # Resolve input tensor names (use actual variable names if possible)
        in_vars: List[str] = []
        for inp in list(getattr(layer, "inputs", []) or []):
            v = _try_call(layer.get_input_variable, inp, default=None)
            in_vars.append(str(_try_getattr(v, "name", inp)))

        out_var = _primary_output_var(layer)
        layer_out_var[layer_name] = out_var

        # Activation
        if kind == "Activation":
            activation = str(layer.attributes.get("activation", "")).lower()
            if activation == "relu":
                op = S.ROperator_Relu[options.tensor_dtype](in_vars[0], out_var)
            elif activation == "elu":
                alpha = float(layer.attributes.get("alpha", 1.0))
                op = S.ROperator_Elu[options.tensor_dtype](alpha, in_vars[0], out_var)
            else:
                if options.strict:
                    raise NotImplementedError(f"Unsupported Activation '{activation}' in layer '{layer_name}'")
                continue
            rmodel.AddOperatorReference(op)
            ROOT.SetOwnership(op, False)
            continue

        # Dense-like (Dense or backend-lowered PointwiseConv1D)
        if kind in {"Dense", "PointwiseConv1D"}:
            attrs = layer.attributes
            wvar = attrs.get("weight")
            bvar = attrs.get("bias")
            if wvar is None or bvar is None or not hasattr(wvar, "data") or not hasattr(bvar, "data"):
                raise RuntimeError(f"Layer '{layer_name}' is missing weight/bias data")

            w = wvar.data
            b = bvar.data
            # Dense: typically (n_in, n_out); PointwiseConv1D: (1, n_in, n_out)
            if hasattr(w, "ndim") and int(getattr(w, "ndim")) == 3 and int(w.shape[0]) == 1:
                w = w.reshape((int(w.shape[1]), int(w.shape[2])))

            w_name = f"{layer_name}_W"
            b_name = f"{layer_name}_B"
            _add_f32_tensor(w_name, [int(w.shape[0]), int(w.shape[1])], w.reshape(-1).tolist())
            _add_f32_tensor(b_name, [int(b.shape[0])], b.reshape(-1).tolist())

            op = S.ROperator_Gemm[options.tensor_dtype](1.0, 1.0, 0, 0, in_vars[0], w_name, b_name, out_var)
            rmodel.AddOperatorReference(op)
            ROOT.SetOwnership(op, False)
            continue

        # Reshape
        if kind == "Reshape":
            target_shape = layer.attributes.get("target_shape")
            if target_shape is None:
                raise RuntimeError(f"Reshape layer '{layer_name}' has no target_shape")
            # hls4ml uses shapes without batch dim; SOFIE Reshape expects int64 values
            shape_tensor_name = f"{layer_name}_shape"
            vals = [int(x) for x in target_shape]
            _add_i64_tensor(shape_tensor_name, [len(vals)], vals)
            # mark as non-writable: the graph constructor effectively hard-codes it
            rmodel.SetNotWritableInitializedTensor(shape_tensor_name)

            op = S.ROperator_Reshape(S.ReshapeOpMode.Reshape, 0, in_vars[0], shape_tensor_name, out_var)
            rmodel.AddOperatorReference(op)
            ROOT.SetOwnership(op, False)
            continue

        # Concatenate (2-input)
        if kind == "Concatenate":
            axis = int(layer.attributes.get("axis", 0))
            # Match hls4ml behavior: axis includes batch dim; internal reshape/concat uses axis-1 for data dims
            axis_eff = axis - 1 if axis > 0 else axis
            op = S.ROperator_Concat(in_vars, axis_eff, 0, out_var)
            rmodel.AddOperatorReference(op)
            ROOT.SetOwnership(op, False)
            continue

        if options.strict:
            raise NotImplementedError(f"Unsupported layer '{layer_name}' of kind '{kind}'")

    # Outputs: use graph outputs if available, else last layer output
    out_names = [str(v.name) for v in modelgraph.get_output_variables()] or ([out_var] if "out_var" in locals() else [])
    if out_names:
        rmodel.AddOutputTensorNameList(out_names)

    return rmodel
