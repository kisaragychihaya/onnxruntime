"""Route A real-GPU validation: WebGPU EP over Dawn's Vulkan backend vs CPU EP.

Prereqs:
    pip install <CI-built wheel>.whl "onnx>=1.17" numpy

Usage:
    python vulkan_local_test.py [--iters 50]

Notes:
- Forces the Vulkan backend via the session config entry
  "ep.webgpuexecutionprovider.dawnBackendType" = "Vulkan".
- ORT verbose logging is enabled: check the console for
  "WebGPU EP Dawn backend type: 6" (6 == WGPUBackendType_Vulkan).
- Unlike the CI lavapipe run, this model exercises the MatMul family, which the
  software Vulkan adapter could not run.
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

OPSET = 20


def build_model(path: Path) -> None:
    rng = np.random.default_rng(0)
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])

    def init(name, arr):
        return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.flatten())

    inits = [
        init("W1", rng.normal(0, 0.05, (64, 128)).astype(np.float32)),
        init("b1", rng.normal(0, 0.02, (128,)).astype(np.float32)),
        init("W2", rng.normal(0, 0.05, (128, 64)).astype(np.float32)),
        init("b2", rng.normal(0, 0.02, (64,)).astype(np.float32)),
    ]

    nodes = [
        helper.make_node("MatMul", ["X", "W1"], ["MM1"]),
        helper.make_node("Add", ["MM1", "b1"], ["A1"]),
        helper.make_node("LayerNormalization", ["A1"], ["LN"], axis=-1, epsilon=1e-5),
        helper.make_node("Gelu", ["LN"], ["GELU"], approximate="tanh"),
        helper.make_node("MatMul", ["GELU", "W2"], ["MM2"]),
        helper.make_node("Add", ["MM2", "b2"], ["A2"]),
        helper.make_node("Softmax", ["A2"], ["Y"], axis=-1),
    ]

    graph = helper.make_graph(nodes, "vulkan_route_a_validation", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_session(model_path: Path, providers):
    so = ort.SessionOptions()
    so.log_severity_level = 0  # VERBOSE: prints "WebGPU EP Dawn backend type: ..."
    so.log_verbosity_level = 0
    so.add_session_config_entry("ep.webgpuexecutionprovider.dawnBackendType", "Vulkan")
    return ort.InferenceSession(str(model_path), sess_options=so, providers=providers)


def benchmark(sess, feed, iters):
    for _ in range(3):
        sess.run(None, feed)
    start = time.perf_counter()
    for _ in range(iters):
        sess.run(None, feed)
    return (time.perf_counter() - start) / iters * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    print("onnxruntime:", ort.__version__)
    print("available providers:", ort.get_available_providers())
    if "WebGpuExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("WebGpuExecutionProvider is NOT available in this build!")

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.onnx"
        build_model(model_path)

        x = np.random.default_rng(1).normal(0, 1, (1, 16, 64)).astype(np.float32)
        feed = {"X": x}

        cpu = make_session(model_path, ["CPUExecutionProvider"])
        out_cpu = cpu.run(None, feed)[0]

        webgpu = make_session(model_path, ["WebGpuExecutionProvider"])
        out_gpu = webgpu.run(None, feed)[0]

        diff = np.max(np.abs(out_cpu - out_gpu))
        ok = np.allclose(out_cpu, out_gpu, rtol=1e-3, atol=1e-3)
        print(f"\nmax |CPU - WebGPU(Vulkan)| = {diff:.3e}")
        print("numeric check:", "PASS" if ok else "FAIL")
        if not ok:
            raise SystemExit(1)

        t_cpu = benchmark(cpu, feed, args.iters)
        t_gpu = benchmark(webgpu, feed, args.iters)
        print(f"\navg latency over {args.iters} iters (input [1,16,64]):")
        print(f"  CPU             : {t_cpu:.3f} ms")
        print(f"  WebGPU (Vulkan) : {t_gpu:.3f} ms")
        print("\nCheck the verbose log above for 'WebGPU EP Dawn backend type: 6' (Vulkan).")


if __name__ == "__main__":
    main()
