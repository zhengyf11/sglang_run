#!/usr/bin/env python3
"""Local Web UI that generates SGLang prefill launch commands without executing them."""

from __future__ import annotations

import argparse
import json
import mimetypes
import shlex
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

DEFAULT_WEB_PORT = 6060
WEB_DIR = Path(__file__).with_name("web")

DEFAULTS: dict[str, Any] = {
    "model_path": "/mnt/GLM-5.1-FP8",
    "served_model_name": "GLM-5.1-FP8",
    "parallel_tp_size": 8,
    "attention_parallel_mode": "tensor",
    "context_parallel_backend": "nsa",
    "moe_parallel_mode": "tensor",
    "enable_single_batch_overlap": False,
    "enable_two_batch_overlap": False,
    "tool_call_parser": "glm47",
    "reasoning_parser": "glm45",
    "enable_mtp": False,
    "speculative_algorithm": "EAGLE",
    "speculative_num_steps": 3,
    "speculative_eagle_topk": 1,
    "speculative_num_draft_tokens": 4,
    "mem_fraction_static": 0.7,
    "host": "0.0.0.0",
    "port": 30000,
    "nnodes": 1,
    "node_rank": 0,
    "dist_init_addr": "127.0.0.1:20000",
    "disaggregation_mode": "prefill",
    "disaggregation_transfer_backend": "mooncake",
    "disaggregation_ib_device": "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7",
    "trust_remote_code": True,
    "disable_cuda_graph": True,
    "max_running_requests": 128,
    "chunked_prefill_size": 8192,
    "max_prefill_tokens": 65536,
    "extra_sglang_args": "",
}

NCCL_ENV_DEFAULTS = {
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_HCA": "^mlx5_bond",
    "NCCL_SOCKET_IFNAME": "bond0",
    "NCCL_IB_TC": "128",
    "NCCL_IB_TIMEOUT": "22",
    "NCCL_IB_RETRY_CNT": "15",
}

PROXY_ENV_VARS = (
    "http_proxy", "https_proxy", "ftp_proxy", "all_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY",
)

MTP_COMMAND_FIELDS: tuple[tuple[str, str], ...] = (
    ("speculative_algorithm", "--speculative-algorithm"),
    ("speculative_num_steps", "--speculative-num-steps"),
    ("speculative_eagle_topk", "--speculative-eagle-topk"),
    ("speculative_num_draft_tokens", "--speculative-num-draft-tokens"),
)

COMMAND_FIELDS: tuple[tuple[str, str], ...] = (
    ("model_path", "--model-path"),
    ("served_model_name", "--served-model-name"),
    ("tool_call_parser", "--tool-call-parser"),
    ("reasoning_parser", "--reasoning-parser"),
    ("mem_fraction_static", "--mem-fraction-static"),
    ("host", "--host"),
    ("port", "--port"),
    ("nnodes", "--nnodes"),
    ("node_rank", "--node-rank"),
    ("dist_init_addr", "--dist-init-addr"),
    ("disaggregation_mode", "--disaggregation-mode"),
    ("disaggregation_transfer_backend", "--disaggregation-transfer-backend"),
    ("disaggregation_ib_device", "--disaggregation-ib-device"),
    ("max_running_requests", "--max-running-requests"),
    ("chunked_prefill_size", "--chunked-prefill-size"),
    ("max_prefill_tokens", "--max-prefill-tokens"),
)


def _has_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and value.strip() == "")


def _to_bool(value: Any, default: bool) -> bool:
    if not _has_value(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def normalize_form_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply Issue #2 defaults when fields are omitted, blank, or null."""
    raw = {} if payload is None else dict(payload)
    config = {key: (raw[key] if _has_value(raw.get(key)) else value) for key, value in DEFAULTS.items()}
    config["enable_mtp"] = _to_bool(raw.get("enable_mtp"), DEFAULTS["enable_mtp"])
    config["enable_single_batch_overlap"] = _to_bool(
        raw.get("enable_single_batch_overlap"), DEFAULTS["enable_single_batch_overlap"]
    )
    config["enable_two_batch_overlap"] = _to_bool(
        raw.get("enable_two_batch_overlap"), DEFAULTS["enable_two_batch_overlap"]
    )
    config["trust_remote_code"] = _to_bool(raw.get("trust_remote_code"), DEFAULTS["trust_remote_code"])
    config["disable_cuda_graph"] = _to_bool(raw.get("disable_cuda_graph"), DEFAULTS["disable_cuda_graph"])
    for env_key, default in NCCL_ENV_DEFAULTS.items():
        config[env_key] = raw[env_key] if _has_value(raw.get(env_key)) else default
    return config


def parse_extra_sglang_args(value: Any) -> list[str]:
    """Parse optional extra SGLang args from shell-like text or a list of strings."""
    if not _has_value(value):
        return []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    if not isinstance(value, str):
        raise ValueError("extra_sglang_args must be a string or a list of strings")
    return shlex.split(value)


def _append_unique_flag(cmd: list[str], flag: str) -> None:
    if flag not in cmd:
        cmd.append(flag)


def build_parallel_args(config: Mapping[str, Any]) -> list[str]:
    """Build SGLang parallel arguments from the dedicated Parallel UI group."""
    tp_size = str(config["parallel_tp_size"])
    cmd: list[str] = []

    attention_mode = str(config.get("attention_parallel_mode", "tensor"))
    if attention_mode == "tensor":
        cmd.extend(["--tp-size", tp_size])
    elif attention_mode == "dp_attention":
        cmd.extend(["--tp-size", tp_size, "--dp-size", tp_size, "--enable-dp-attention"])
    elif attention_mode == "context_parallel":
        cmd.extend(["--tp-size", tp_size])
        backend = str(config.get("context_parallel_backend", "nsa"))
        if backend == "prefill":
            cmd.append("--enable-prefill-context-parallel")
        else:
            cmd.append("--enable-nsa-prefill-context-parallel")
        cmd.extend(["--nsa-prefill-cp-mode", "in-seq-split"])
        _append_unique_flag(cmd, "--enable-two-batch-overlap")
    elif attention_mode == "pipeline_parallel":
        cmd.extend(["--tp-size", "1", "--pp-size", tp_size, "--enable-dynamic-chunking"])
    else:
        raise ValueError(f"unsupported attention_parallel_mode: {attention_mode}")

    moe_mode = str(config.get("moe_parallel_mode", "tensor"))
    if moe_mode == "tensor":
        pass
    elif moe_mode == "expert_parallel":
        cmd.append(f"--ep-size={tp_size}")
        cmd.extend(["--moe-a2a-backend", "deepep"])
        if config.get("enable_single_batch_overlap"):
            _append_unique_flag(cmd, "--enable-single-batch-overlap")
        if config.get("enable_two_batch_overlap"):
            _append_unique_flag(cmd, "--enable-two-batch-overlap")
    else:
        raise ValueError(f"unsupported moe_parallel_mode: {moe_mode}")

    return cmd


def build_prefill_command(config: Mapping[str, Any]) -> list[str]:
    """Build a launch_server command list only; this module never executes it."""
    cmd = ["python3", "-m", "sglang.launch_server"]
    for field, flag in COMMAND_FIELDS:
        cmd.extend([flag, str(config[field])])
        if field == "served_model_name":
            cmd.extend(build_parallel_args(config))
    if config.get("enable_mtp"):
        for field, flag in MTP_COMMAND_FIELDS:
            cmd.extend([flag, str(config[field])])
    if config.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if config.get("disable_cuda_graph"):
        cmd.append("--disable-cuda-graph")
    cmd.extend(parse_extra_sglang_args(config.get("extra_sglang_args")))
    return cmd


def build_shell_command(command: Sequence[str]) -> str:
    if len(command) <= 3:
        return shlex.join(command)

    head = shlex.join(command[:3])
    argument_groups: list[Sequence[str]] = []
    index = 3
    while index < len(command):
        current = command[index]
        if current.startswith("--") and index + 1 < len(command) and not command[index + 1].startswith("--"):
            argument_groups.append(command[index:index + 2])
            index += 2
        else:
            argument_groups.append(command[index:index + 1])
            index += 1

    lines = [f"{head} \\"]
    for group_index, group in enumerate(argument_groups):
        suffix = " \\" if group_index < len(argument_groups) - 1 else ""
        lines.append(f"\t{shlex.join(group)}{suffix}")
    return "\n".join(lines)


def build_env_exports(config: Mapping[str, Any]) -> list[str]:
    return [f"export {key}={shlex.quote(str(config[key]))}" for key in NCCL_ENV_DEFAULTS]


def build_shell_hints(config: Mapping[str, Any]) -> list[str]:
    if config.get("attention_parallel_mode") == "pipeline_parallel":
        return ["#export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.8"]
    return []


def build_command_response(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    config = normalize_form_payload(payload)
    command = build_prefill_command(config)
    env_exports = build_env_exports(config)
    shell_hints = build_shell_hints(config)
    proxy_unsets = [f"unset {key}" for key in PROXY_ENV_VARS]
    shell_command = build_shell_command(command)
    return {
        "config": config,
        "command": command,
        "shell_command": shell_command,
        "env_exports": env_exports,
        "shell_hints": shell_hints,
        "proxy_unsets": proxy_unsets,
        "combined_shell": "\n".join([*proxy_unsets, *env_exports, *shell_hints, shell_command]),
        "executed": False,
    }


def read_static_asset(path: str) -> tuple[bytes, str]:
    """Read a Web UI asset from the local web directory."""
    route = "index.html" if path in {"", "/"} else path.lstrip("/")
    if route not in {"index.html", "styles.css", "app.js"}:
        raise FileNotFoundError(path)
    asset_path = WEB_DIR / route
    content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
    if asset_path.suffix == ".js":
        content_type = "application/javascript"
    if asset_path.suffix in {".html", ".css", ".js"}:
        content_type = f"{content_type}; charset=utf-8"
    return asset_path.read_bytes(), content_type


class PrefillCommandHandler(BaseHTTPRequestHandler):
    server_version = "PrefillCommandWeb/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/defaults":
            self._write_json(HTTPStatus.OK, {"defaults": {**DEFAULTS, **NCCL_ENV_DEFAULTS}})
            return
        try:
            body, content_type = read_static_asset(path)
        except FileNotFoundError:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._write_bytes(HTTPStatus.OK, body, content_type)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/command":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            response = build_command_response(payload)
        except json.JSONDecodeError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc.msg}"})
            return
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._write_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _write_json(self, status: HTTPStatus, data: Mapping[str, Any]) -> None:
        self._write(status, json.dumps(data, ensure_ascii=False), "application/json; charset=utf-8")

    def _write(self, status: HTTPStatus, body: str, content_type: str) -> None:
        self._write_bytes(status, body.encode("utf-8"), content_type)

    def _write_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local Web UI that generates SGLang prefill commands.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local command-generator Web UI.")
    parser.add_argument("--port", type=int, default=DEFAULT_WEB_PORT, help="Port for the local command-generator Web UI.")
    return parser.parse_args(argv)


def run_server(host: str, port: int) -> None:
    with ThreadingHTTPServer((host, port), PrefillCommandHandler) as server:
        print(f"Serving prefill command generator at http://{host}:{port}/")
        server.serve_forever()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_server(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
