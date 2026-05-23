#!/usr/bin/env python3
"""Local Web UI that generates SGLang prefill launch commands without executing them."""

from __future__ import annotations

import argparse
import json
import shlex
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence

DEFAULTS: dict[str, Any] = {
    "model_path": "/mnt/GLM-5.1-FP8",
    "served_model_name": "GLM-5.1-FP8",
    "tensor_parallel_size": 8,
    "tool_call_parser": "glm47",
    "reasoning_parser": "glm45",
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

COMMAND_FIELDS: tuple[tuple[str, str], ...] = (
    ("model_path", "--model-path"),
    ("served_model_name", "--served-model-name"),
    ("tensor_parallel_size", "--tensor-parallel-size"),
    ("tool_call_parser", "--tool-call-parser"),
    ("reasoning_parser", "--reasoning-parser"),
    ("speculative_algorithm", "--speculative-algorithm"),
    ("speculative_num_steps", "--speculative-num-steps"),
    ("speculative_eagle_topk", "--speculative-eagle-topk"),
    ("speculative_num_draft_tokens", "--speculative-num-draft-tokens"),
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

FIELD_LABELS = {
    "model_path": "Model path",
    "served_model_name": "Served model name",
    "tensor_parallel_size": "Tensor parallel size",
    "tool_call_parser": "Tool call parser",
    "reasoning_parser": "Reasoning parser",
    "speculative_algorithm": "Speculative algorithm",
    "speculative_num_steps": "Speculative num steps",
    "speculative_eagle_topk": "Speculative EAGLE top-k",
    "speculative_num_draft_tokens": "Speculative draft tokens",
    "mem_fraction_static": "Mem fraction static",
    "host": "SGLang host",
    "port": "SGLang port",
    "nnodes": "Number of nodes",
    "node_rank": "Node rank",
    "dist_init_addr": "Dist init address",
    "disaggregation_mode": "Disaggregation mode",
    "disaggregation_transfer_backend": "Disaggregation backend",
    "disaggregation_ib_device": "Disaggregation IB device",
    "max_running_requests": "Max running requests",
    "chunked_prefill_size": "Chunked prefill size",
    "max_prefill_tokens": "Max prefill tokens",
}


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


def build_prefill_command(config: Mapping[str, Any]) -> list[str]:
    """Build a launch_server command list only; this module never executes it."""
    cmd = ["python3", "-m", "sglang.launch_server"]
    for field, flag in COMMAND_FIELDS:
        cmd.extend([flag, str(config[field])])
    if config.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if config.get("disable_cuda_graph"):
        cmd.append("--disable-cuda-graph")
    cmd.extend(parse_extra_sglang_args(config.get("extra_sglang_args")))
    return cmd


def build_shell_command(command: Sequence[str]) -> str:
    return shlex.join(command)


def build_env_exports(config: Mapping[str, Any]) -> list[str]:
    return [f"export {key}={shlex.quote(str(config[key]))}" for key in NCCL_ENV_DEFAULTS]


def build_command_response(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    config = normalize_form_payload(payload)
    command = build_prefill_command(config)
    env_exports = build_env_exports(config)
    proxy_unsets = [f"unset {key}" for key in PROXY_ENV_VARS]
    shell_command = build_shell_command(command)
    return {
        "config": config,
        "command": command,
        "shell_command": shell_command,
        "env_exports": env_exports,
        "proxy_unsets": proxy_unsets,
        "combined_shell": "\n".join([*proxy_unsets, *env_exports, shell_command]),
        "executed": False,
    }


def render_index_html() -> str:
    defaults_json = json.dumps({**DEFAULTS, **NCCL_ENV_DEFAULTS}, ensure_ascii=False)
    command_inputs = "\n".join(
        f'<label>{FIELD_LABELS[field]}<input name="{field}" placeholder="{DEFAULTS[field]}"></label>'
        for field, _ in COMMAND_FIELDS
    )
    nccl_inputs = "\n".join(
        f'<label>{key}<input name="{key}" placeholder="{value}"></label>'
        for key, value in NCCL_ENV_DEFAULTS.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SGLang Prefill Command Generator</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f7f7fb; color: #1f2937; }}
    main {{ display: grid; grid-template-columns: minmax(320px, 1fr) minmax(360px, 1fr); gap: 1.5rem; }}
    label {{ display: block; margin: .55rem 0; font-weight: 600; }}
    input, textarea {{ box-sizing: border-box; width: 100%; margin-top: .2rem; padding: .45rem; }}
    textarea.output {{ min-height: 18rem; font-family: ui-monospace, monospace; }}
    section {{ background: white; border-radius: .75rem; padding: 1rem; box-shadow: 0 1px 8px #0001; }}
    .checks label {{ display: inline-flex; gap: .4rem; align-items: center; margin-right: 1rem; }}
    .checks input {{ width: auto; }}
    .error {{ color: #b91c1c; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>SGLang Prefill Command Generator</h1>
  <p>This local page generates a prefill shell command only. It does not start SGLang.</p>
  <main>
    <section>
      <h2>Prefill options</h2>
      <form id="command-form">
        {command_inputs}
        <div class="checks">
          <label><input type="checkbox" name="trust_remote_code" checked> trust remote code</label>
          <label><input type="checkbox" name="disable_cuda_graph" checked> disable CUDA graph</label>
        </div>
        <label>Extra SGLang args<textarea name="extra_sglang_args" placeholder="--log-level debug"></textarea></label>
        <h3>NCCL environment exports</h3>
        {nccl_inputs}
      </form>
    </section>
    <section>
      <h2>Generated prefill shell command</h2>
      <textarea id="command-output" class="output" readonly></textarea>
      <h2>Environment exports</h2>
      <textarea id="env-output" class="output" readonly></textarea>
      <p id="error" class="error"></p>
    </section>
  </main>
  <script>
    const defaults = {defaults_json};
    const form = document.getElementById('command-form');
    const commandOutput = document.getElementById('command-output');
    const envOutput = document.getElementById('env-output');
    const errorOutput = document.getElementById('error');
    async function refreshCommand() {{
      const data = {{}};
      for (const element of form.elements) {{
        if (!element.name) continue;
        data[element.name] = element.type === 'checkbox' ? element.checked : element.value;
      }}
      try {{
        const response = await fetch('/api/command', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(data)}});
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || 'Failed to generate command');
        commandOutput.value = body.shell_command;
        envOutput.value = [...body.proxy_unsets, ...body.env_exports].join('\n');
        errorOutput.textContent = '';
      }} catch (error) {{ errorOutput.textContent = error.message; }}
    }}
    form.addEventListener('input', refreshCommand);
    refreshCommand();
  </script>
</body>
</html>"""


class PrefillCommandHandler(BaseHTTPRequestHandler):
    server_version = "PrefillCommandWeb/1.0"

    def do_GET(self) -> None:
        if self.path == "/":
            self._write(HTTPStatus.OK, render_index_html(), "text/html; charset=utf-8")
        elif self.path == "/api/defaults":
            self._write_json(HTTPStatus.OK, {"defaults": {**DEFAULTS, **NCCL_ENV_DEFAULTS}})
        else:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/command":
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
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local Web UI that generates SGLang prefill commands.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local command-generator Web UI.")
    parser.add_argument("--port", type=int, default=8080, help="Port for the local command-generator Web UI.")
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
