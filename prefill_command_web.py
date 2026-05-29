#!/usr/bin/env python3
"""Local Web UI that generates SGLang prefill launch commands without executing them."""

from __future__ import annotations

import argparse
import ast
import json
import mimetypes
import re
import shlex
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

DEFAULT_WEB_PORT = 6060
WEB_DIR = Path(__file__).with_name("web")
SGLANG_SOURCE_DIR = Path(r"D:\AI agent\code\sglang-main-20260519")
TOOL_CALL_PARSER_SOURCE = SGLANG_SOURCE_DIR / "python" / "sglang" / "srt" / "function_call" / "function_call_parser.py"
REASONING_PARSER_SOURCE = SGLANG_SOURCE_DIR / "python" / "sglang" / "srt" / "parser" / "reasoning_parser.py"

UNKNOWN_PARSER = "unknown"

PREFILL_PROFILE = "prefill"
DECODE_PROFILE = "decode"
ROUTER_PROFILE = "router"
SUPPORTED_PROFILES = {PREFILL_PROFILE, DECODE_PROFILE, ROUTER_PROFILE}

DEFAULTS: dict[str, Any] = {
    "model_path": "/mnt/GLM-5.1-FP8",
    "served_model_name": "GLM-5.1-FP8",
    "parallel_tp_size": 8,
    "dp_size": 8,
    "attention_parallel_mode": "tensor",
    "context_parallel_backend": "nsa",
    "enable_dynamic_chunking": True,
    "dynamic_chunking_smooth_factor": "0.8",
    "moe_parallel_mode": "tensor",
    "enable_single_batch_overlap": False,
    "enable_two_batch_overlap": False,
    "tool_call_parser": UNKNOWN_PARSER,
    "reasoning_parser": UNKNOWN_PARSER,
    "enable_mtp": False,
    "speculative_algorithm": "EAGLE",
    "speculative_num_steps": 3,
    "speculative_eagle_topk": 1,
    "speculative_num_draft_tokens": 4,
    "mem_fraction_static": 0.9,
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

DECODE_DEFAULTS: dict[str, Any] = {
    **DEFAULTS,
    "enable_mtp": True,
    "mem_fraction_static": 0.7,
    "port": 30001,
    "dist_init_addr": "192.168.1.233:20000",
    "disaggregation_mode": "decode",
    "disable_cuda_graph": False,
}

ROUTER_DEFAULTS: dict[str, Any] = {
    "model_path": "/mnt/GLM-5.1-FP8",
    "served_model_name": "GLM-5.1-FP8",
    "tool_call_parser": "glm47_moe",
    "reasoning_parser": "glm45",
    "prefill": "http://192.168.1.99:30000",
    "decode": "http://192.168.1.233:30001",
    "host": "0.0.0.0",
    "port": 8000,
    "policy": "cache_aware",
    "retry_max_retries": 3,
    "extra_router_args": "",
}

PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    PREFILL_PROFILE: DEFAULTS,
    DECODE_PROFILE: DECODE_DEFAULTS,
    ROUTER_PROFILE: ROUTER_DEFAULTS,
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

FIXED_FORM_DEFAULT_KEYS = {
    "nnodes",
    "node_rank",
    "disaggregation_mode",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TC",
    "NCCL_IB_TIMEOUT",
    "NCCL_IB_RETRY_CNT",
}

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

ROUTER_COMMAND_FIELDS: tuple[tuple[str, str], ...] = (
    ("prefill", "--prefill"),
    ("decode", "--decode"),
    ("host", "--host"),
    ("port", "--port"),
    ("policy", "--policy"),
    ("model_path", "--model-path"),
    ("served_model_name", "--served-model-name"),
    ("tool_call_parser", "--tool-call-parser"),
    ("reasoning_parser", "--reasoning-parser"),
    ("retry_max_retries", "--retry-max-retries"),
)

PATH_SEGMENTS_TO_IGNORE = {
    "mnt",
    "mount",
    "model",
    "models",
    "vllm",
    "sglang",
    "workspace",
    "data",
    "api",
}

TOOL_CALL_PARSER_FALLBACK_CHOICES = (
    UNKNOWN_PARSER,
    "deepseekv3", "deepseekv31", "deepseekv32", "deepseekv4", "glm", "glm45", "glm47", "glm47_moe",
    "gpt-oss", "kimi_k2", "lfm2", "llama3", "mimo", "mistral", "poolside_v1", "pythonic",
    "qwen", "qwen25", "qwen3_coder", "step3", "step3p5", "minimax-m2", "trinity",
    "interns1", "hermes", "hunyuan", "gigachat3", "gemma4",
)
REASONING_PARSER_FALLBACK_CHOICES = (
    UNKNOWN_PARSER,
    "deepseek-r1", "deepseek-v3", "deepseek-v4", "glm45", "hunyuan", "gpt-oss", "kimi",
    "kimi_k2", "mimo", "poolside_v1", "qwen3", "qwen3-thinking", "minimax",
    "minimax-append-think", "step3", "step3p5", "mistral", "nemotron_3", "interns1", "gemma4",
)

PARSER_INFERENCE_RULES: tuple[dict[str, Any], ...] = (
    {"patterns": ("qwen3-coder",), "tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3"},
    {"patterns": ("qwen3", "qwen-3", "qwen_3"), "tool_call_parser": "qwen", "reasoning_parser": "qwen3"},
    {"patterns": ("qwen2.5", "qwen25", "qwen-2.5"), "tool_call_parser": "qwen25", "reasoning_parser": "qwen3"},
    {"patterns": ("qwen",), "tool_call_parser": "qwen", "reasoning_parser": "qwen3"},
    {"patterns": ("deepseek-v3.2", "deepseekv3.2", "deepseek-v32", "deepseekv32"), "tool_call_parser": "deepseekv32", "reasoning_parser": "deepseek-v3"},
    {"patterns": ("deepseek-v3.1", "deepseekv3.1", "deepseek-v31", "deepseekv31"), "tool_call_parser": "deepseekv31", "reasoning_parser": "deepseek-v3"},
    {"patterns": ("deepseek-v4", "deepseekv4"), "tool_call_parser": "deepseekv4", "reasoning_parser": "deepseek-v4"},
    {"patterns": ("deepseek-r1",), "tool_call_parser": "deepseekv3", "reasoning_parser": "deepseek-r1"},
    {"patterns": ("deepseek-v3", "deepseekv3"), "tool_call_parser": "deepseekv3", "reasoning_parser": "deepseek-v3"},
    {"patterns": ("glm",), "tool_call_parser": "glm47", "reasoning_parser": "glm45"},
    {"patterns": ("minimax-m2", "minimax"), "tool_call_parser": "minimax-m2", "reasoning_parser": "minimax"},
    {"patterns": ("kimi-k2", "kimi_k2"), "tool_call_parser": "kimi_k2", "reasoning_parser": "kimi_k2"},
    {"patterns": ("kimi",), "tool_call_parser": "kimi_k2", "reasoning_parser": "kimi"},
    {"patterns": ("gpt-oss",), "tool_call_parser": "gpt-oss", "reasoning_parser": "gpt-oss"},
    {"patterns": ("mistral",), "tool_call_parser": "mistral", "reasoning_parser": "mistral"},
    {"patterns": ("step3p5", "step-3.5", "step3.5"), "tool_call_parser": "step3p5", "reasoning_parser": "step3p5"},
    {"patterns": ("step3", "step-3"), "tool_call_parser": "step3", "reasoning_parser": "step3"},
    {"patterns": ("hunyuan",), "tool_call_parser": "hunyuan", "reasoning_parser": "hunyuan"},
    {"patterns": ("gemma",), "tool_call_parser": "gemma4", "reasoning_parser": "gemma4"},
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


def _extract_dict_keys_from_python(path: Path, assignment_name: str) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ()

    keys: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        target = node.target
        if not isinstance(target, ast.Name) or target.id != assignment_name or not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
    return tuple(dict.fromkeys(keys))


def _merge_parser_choices(primary: Sequence[str], fallback: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*primary, *fallback]))


def get_tool_call_parser_choices() -> tuple[str, ...]:
    choices = _extract_dict_keys_from_python(TOOL_CALL_PARSER_SOURCE, "ToolCallParserEnum")
    return _merge_parser_choices(choices, TOOL_CALL_PARSER_FALLBACK_CHOICES)


def get_reasoning_parser_choices() -> tuple[str, ...]:
    choices = _extract_dict_keys_from_python(REASONING_PARSER_SOURCE, "DetectorMap")
    return _merge_parser_choices(choices, REASONING_PARSER_FALLBACK_CHOICES)


def _is_ignored_path_segment(segment: str) -> bool:
    normalized = segment.strip().lower()
    return normalized in PATH_SEGMENTS_TO_IGNORE or re.fullmatch(r"v\d+", normalized) is not None


def infer_served_model_name(model_path: Any) -> str:
    if not _has_value(model_path):
        return DEFAULTS["served_model_name"]
    parts = [part for part in str(model_path).replace("\\", "/").split("/") if part]
    for part in reversed(parts):
        if not _is_ignored_path_segment(part):
            return part
    return DEFAULTS["served_model_name"]


def _choose_parser(candidate: str, choices: Sequence[str], fallback: str) -> str:
    return candidate if candidate in choices else fallback


def infer_model_parsers(model_path: Any) -> dict[str, str]:
    model_name = infer_served_model_name(model_path)
    haystack = f"{model_path or ''} {model_name}".lower().replace("_", "-")
    tool_choices = get_tool_call_parser_choices()
    reasoning_choices = get_reasoning_parser_choices()
    tool_fallback = DEFAULTS["tool_call_parser"]
    reasoning_fallback = DEFAULTS["reasoning_parser"]

    for rule in PARSER_INFERENCE_RULES:
        if any(pattern in haystack for pattern in rule["patterns"]):
            return {
                "tool_call_parser": _choose_parser(rule["tool_call_parser"], tool_choices, tool_fallback),
                "reasoning_parser": _choose_parser(rule["reasoning_parser"], reasoning_choices, reasoning_fallback),
            }
    return {"tool_call_parser": tool_fallback, "reasoning_parser": reasoning_fallback}


def get_parser_metadata() -> dict[str, Any]:
    return {
        "tool_call_parser_choices": list(get_tool_call_parser_choices()),
        "reasoning_parser_choices": list(get_reasoning_parser_choices()),
        "fallbacks": {
            "tool_call_parser": DEFAULTS["tool_call_parser"],
            "reasoning_parser": DEFAULTS["reasoning_parser"],
        },
        "rules": [
            {
                "patterns": list(rule["patterns"]),
                "tool_call_parser": rule["tool_call_parser"],
                "reasoning_parser": rule["reasoning_parser"],
            }
            for rule in PARSER_INFERENCE_RULES
        ],
        "ignored_model_path_segments": sorted(PATH_SEGMENTS_TO_IGNORE),
    }


def normalize_profile(profile: Any) -> str:
    if not _has_value(profile):
        return PREFILL_PROFILE
    normalized = str(profile).strip().lower()
    if normalized not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported profile: {profile}")
    return normalized


def get_profile_defaults(profile: str = PREFILL_PROFILE) -> dict[str, Any]:
    return PROFILE_DEFAULTS[normalize_profile(profile)]


def get_effective_defaults(profile: str = PREFILL_PROFILE) -> dict[str, Any]:
    """Return UI/API defaults after applying model-path-derived parser values."""
    normalized_profile = normalize_profile(profile)
    base_defaults = get_profile_defaults(normalized_profile)
    defaults = dict(base_defaults)
    if normalized_profile in {PREFILL_PROFILE, DECODE_PROFILE}:
        defaults.update(NCCL_ENV_DEFAULTS)
    defaults.update(infer_model_parsers(defaults["model_path"]))
    if normalized_profile == ROUTER_PROFILE:
        defaults["tool_call_parser"] = base_defaults["tool_call_parser"]
    defaults["served_model_name"] = infer_served_model_name(defaults["model_path"])
    return defaults


def normalize_form_payload(payload: Mapping[str, Any] | None, profile: str = PREFILL_PROFILE) -> dict[str, Any]:
    """Apply profile defaults when fields are omitted, blank, or null."""
    normalized_profile = normalize_profile(profile)
    defaults = get_profile_defaults(normalized_profile)
    raw = {} if payload is None else dict(payload)
    fixed_keys = set(FIXED_FORM_DEFAULT_KEYS)
    config = {
        key: (raw[key] if key not in fixed_keys and _has_value(raw.get(key)) else value)
        for key, value in defaults.items()
    }

    if _has_value(raw.get("model_path")) and not _has_value(raw.get("served_model_name")):
        config["served_model_name"] = infer_served_model_name(raw["model_path"])

    inferred_parsers = infer_model_parsers(config["model_path"])
    tool_choices = get_tool_call_parser_choices()
    reasoning_choices = get_reasoning_parser_choices()
    config["tool_call_parser"] = (
        raw["tool_call_parser"]
        if _has_value(raw.get("tool_call_parser")) and raw["tool_call_parser"] in tool_choices
        else inferred_parsers["tool_call_parser"]
    )
    config["reasoning_parser"] = (
        raw["reasoning_parser"]
        if _has_value(raw.get("reasoning_parser")) and raw["reasoning_parser"] in reasoning_choices
        else inferred_parsers["reasoning_parser"]
    )
    if normalized_profile == ROUTER_PROFILE and not _has_value(raw.get("tool_call_parser")):
        config["tool_call_parser"] = defaults["tool_call_parser"]

    for bool_key in (
        "enable_mtp",
        "enable_dynamic_chunking",
        "enable_single_batch_overlap",
        "enable_two_batch_overlap",
        "trust_remote_code",
        "disable_cuda_graph",
    ):
        if bool_key in defaults:
            config[bool_key] = _to_bool(raw.get(bool_key), defaults[bool_key])

    if normalized_profile in {PREFILL_PROFILE, DECODE_PROFILE}:
        for env_key, default in NCCL_ENV_DEFAULTS.items():
            config[env_key] = default if env_key in fixed_keys else raw[env_key] if _has_value(raw.get(env_key)) else default
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
        dp_size = str(config.get("dp_size") if _has_value(config.get("dp_size")) else config["parallel_tp_size"])
        cmd.extend(["--tp-size", tp_size, "--dp-size", dp_size, "--enable-dp-attention"])
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
        cmd.extend(["--tp-size", "1", "--pp-size", tp_size])
        if config.get("enable_dynamic_chunking"):
            cmd.append("--enable-dynamic-chunking")
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


def build_router_command(config: Mapping[str, Any]) -> list[str]:
    """Build a launch_router command list only; this module never executes it."""
    cmd = ["python3", "-m", "sglang_router.launch_router", "--pd-disaggregation"]
    for field, flag in ROUTER_COMMAND_FIELDS:
        cmd.extend([flag, str(config[field])])
    cmd.extend(parse_extra_sglang_args(config.get("extra_router_args")))
    return cmd


def build_profile_command(profile: str, config: Mapping[str, Any]) -> list[str]:
    normalized_profile = normalize_profile(profile)
    if normalized_profile == ROUTER_PROFILE:
        return build_router_command(config)
    return build_prefill_command(config)


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
    if config.get("attention_parallel_mode") == "pipeline_parallel" and config.get("enable_dynamic_chunking"):
        smooth_factor = config.get("dynamic_chunking_smooth_factor", DEFAULTS["dynamic_chunking_smooth_factor"])
        return [f"#export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR={smooth_factor}"]
    return []


def build_resource_limits(profile: str) -> list[str]:
    return ["ulimit -l unlimited", "ulimit -n 65535"] if normalize_profile(profile) == DECODE_PROFILE else []


def build_command_response(payload: Mapping[str, Any] | None, profile: str = PREFILL_PROFILE) -> dict[str, Any]:
    normalized_profile = normalize_profile(profile)
    config = normalize_form_payload(payload, normalized_profile)
    command = build_profile_command(normalized_profile, config)
    resource_limits = build_resource_limits(normalized_profile)
    env_exports = build_env_exports(config) if normalized_profile in {PREFILL_PROFILE, DECODE_PROFILE} else []
    shell_hints = build_shell_hints(config) if normalized_profile in {PREFILL_PROFILE, DECODE_PROFILE} else []
    proxy_unsets = [f"unset {key}" for key in PROXY_ENV_VARS]
    shell_command = build_shell_command(command)
    return {
        "profile": normalized_profile,
        "config": config,
        "command": command,
        "shell_command": shell_command,
        "resource_limits": resource_limits,
        "env_exports": env_exports,
        "shell_hints": shell_hints,
        "proxy_unsets": proxy_unsets,
        "combined_shell": "\n".join([*resource_limits, *proxy_unsets, *env_exports, *shell_hints, shell_command]),
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
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        if path == "/api/defaults":
            try:
                query = dict(part.split("=", 1) for part in parsed_url.query.split("&") if part and "=" in part)
                profile = normalize_profile(query.get("profile"))
                self._write_json(
                    HTTPStatus.OK,
                    {"profile": profile, "defaults": get_effective_defaults(profile), "parser_metadata": get_parser_metadata()},
                )
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
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
            profile = normalize_profile(payload.pop("profile", PREFILL_PROFILE))
            response = build_command_response(payload, profile)
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
