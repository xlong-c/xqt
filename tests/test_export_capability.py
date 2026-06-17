from xqt.export import DEFAULT_EXPORT_CAPABILITIES, deployment_capability_matrix


def test_deployment_capability_matrix_contains_core_formats() -> None:
    formats = {record.format for record in DEFAULT_EXPORT_CAPABILITIES}

    assert {
        "torch_export",
        "torchscript",
        "onnx",
        "tensorrt",
        "openvino",
        "executorch",
        "ncnn",
        "mnn",
    } <= formats


def test_deployment_capability_matrix_filters_priority_and_status() -> None:
    p0 = deployment_capability_matrix(priority="P0")
    implemented = deployment_capability_matrix(implemented_only=True)

    assert {record.format for record in p0} == {"torch_export", "onnx", "tensorrt"}
    assert {record.format for record in implemented} >= {
        "torch_export",
        "torchscript",
        "onnx",
        "tensorrt",
        "openvino",
    }
    assert all(record.status in {"implemented", "adapter"} for record in implemented)
    assert implemented[0].to_dict()["runtimes"]
