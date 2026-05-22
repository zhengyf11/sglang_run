#!/usr/bin/env python3
"""Run an SGLang disaggregated prefill process."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import Mapping, Sequence

DEFAULT_MODEL_PATH = "/mnt/GLM-5.1-FP8"
DEFAULT_SERVED_MODEL_NAME = "GLM-5.1-FP8"
DEFAULT_TENSOR_PARALLEL_SIZE = 8
DEFAULT_TOOL_CALL_PARSER = "glm47"
DEFAULT_REASONING_PARSER = "glm45"
DEFAULT_SPECULATIVE_ALGORITHM = "EAGLE"
DEFAULT_SPECULATIVE_NUM_STEPS = 3
DEFAULT_SPECULATIVE_EAGLE_TOPK = 1
DEFAULT_SPECULATIVE_NUM_DRAFT_TOKENS = 4
DEFAULT_MEM_FRACTION_STATIC = 0.7
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 30000
DEFAULT_NNODES = 1
DEFAULT_NODE_RANK = 0
DEFAULT_DIST_INIT_ADDR = "127.0.0.1:20000"
DEFAULT_DISAGGREGATION_MODE = "prefill"
DEFAULT_DISAGGREGATION_TRANSFER_BACKEND = "mooncake"
DEFAULT_DISAGGREGATION_IB_DEVICE = (
    "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,"
    "mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7"
)
DEFAULT_MAX_RUNNING_REQUESTS = 128
DEFAULT_CHUNKED_PREFILL_SIZE = 8192
DEFAULT_MAX_PREFILL_TOKENS = 65536

PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "ftp_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "FTP_PROXY",
    "ALL_PROXY",
)

NCCL_ENV_DEFAULTS = {
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_HCA": "^mlx5_bond",
    "NCCL_SOCKET_IFNAME": "bond0",
    "NCCL_IB_TC": "128",
    "NCCL_IB_TIMEOUT": "22",
    "NCCL_IB_RETRY_CNT": "15",
}


def normalize_extra_arg_options(argv: Sequence[str] | None) -> list[str] | None:
    """Allow --sglang-arg values that start with a dash.

    argparse treats a value such as ``--log-level`` after ``--sglang-arg`` as a
    new option instead of as that option's value. Converting this passthrough
    option to ``--sglang-arg=value`` preserves the standard user-facing
    ``--key value`` form while allowing raw SGLang flags.
    """
    raw_argv = sys.argv[1:] if argv is None else list(argv)

    normalized: list[str] = []
    index = 0
    while index < len(raw_argv):
        item = raw_argv[index]
        if item == "--sglang-arg" and index + 1 < len(raw_argv):
            normalized.append(f"{item}={raw_argv[index + 1]}")
            index += 2
            continue
        normalized.append(item)
        index += 1
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the prefill runner."""
    parser = argparse.ArgumentParser(
        description="Run an SGLang disaggregated prefill launch_server process."
    )

    server_group = parser.add_argument_group("SGLang launch_server options")
    server_group.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    server_group.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL_NAME)
    server_group.add_argument(
        "--tensor-parallel-size",
        "--tp",
        dest="tensor_parallel_size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help="Tensor parallel size passed as --tensor-parallel-size.",
    )
    server_group.add_argument("--tool-call-parser", default=DEFAULT_TOOL_CALL_PARSER)
    server_group.add_argument("--reasoning-parser", default=DEFAULT_REASONING_PARSER)
    server_group.add_argument("--speculative-algorithm", default=DEFAULT_SPECULATIVE_ALGORITHM)
    server_group.add_argument("--speculative-num-steps", type=int, default=DEFAULT_SPECULATIVE_NUM_STEPS)
    server_group.add_argument("--speculative-eagle-topk", type=int, default=DEFAULT_SPECULATIVE_EAGLE_TOPK)
    server_group.add_argument(
        "--speculative-num-draft-tokens",
        type=int,
        default=DEFAULT_SPECULATIVE_NUM_DRAFT_TOKENS,
    )
    server_group.add_argument("--mem-fraction-static", type=float, default=DEFAULT_MEM_FRACTION_STATIC)
    server_group.add_argument("--host", default=DEFAULT_HOST)
    server_group.add_argument("--port", type=int, default=DEFAULT_PORT)
    server_group.add_argument("--nnodes", type=int, default=DEFAULT_NNODES)
    server_group.add_argument("--node-rank", type=int, default=DEFAULT_NODE_RANK)
    server_group.add_argument("--dist-init-addr", default=DEFAULT_DIST_INIT_ADDR)
    server_group.add_argument("--disaggregation-mode", default=DEFAULT_DISAGGREGATION_MODE)
    server_group.add_argument(
        "--disaggregation-transfer-backend",
        default=DEFAULT_DISAGGREGATION_TRANSFER_BACKEND,
    )
    server_group.add_argument("--disaggregation-ib-device", default=DEFAULT_DISAGGREGATION_IB_DEVICE)
    server_group.add_argument("--max-running-requests", type=int, default=DEFAULT_MAX_RUNNING_REQUESTS)
    server_group.add_argument("--chunked-prefill-size", type=int, default=DEFAULT_CHUNKED_PREFILL_SIZE)
    server_group.add_argument("--max-prefill-tokens", type=int, default=DEFAULT_MAX_PREFILL_TOKENS)
    server_group.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --trust-remote-code by default; use --no-trust-remote-code to omit it.",
    )
    server_group.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        default=True,
        help="Pass --disable-cuda-graph by default.",
    )
    server_group.add_argument(
        "--enable-cuda-graph",
        dest="disable_cuda_graph",
        action="store_false",
        help="Omit --disable-cuda-graph from the SGLang command.",
    )
    server_group.add_argument(
        "--sglang-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra raw SGLang launch_server argument appended after built-in options. Can be repeated.",
    )

    nccl_group = parser.add_argument_group("NCCL network environment")
    nccl_group.add_argument("--nccl-ib-gid-index", default=NCCL_ENV_DEFAULTS["NCCL_IB_GID_INDEX"])
    nccl_group.add_argument("--nccl-ib-hca", default=NCCL_ENV_DEFAULTS["NCCL_IB_HCA"])
    nccl_group.add_argument("--nccl-socket-ifname", default=NCCL_ENV_DEFAULTS["NCCL_SOCKET_IFNAME"])
    nccl_group.add_argument("--nccl-ib-tc", default=NCCL_ENV_DEFAULTS["NCCL_IB_TC"])
    nccl_group.add_argument("--nccl-ib-timeout", default=NCCL_ENV_DEFAULTS["NCCL_IB_TIMEOUT"])
    nccl_group.add_argument("--nccl-ib-retry-cnt", default=NCCL_ENV_DEFAULTS["NCCL_IB_RETRY_CNT"])

    runtime_group = parser.add_argument_group("Runtime options")
    runtime_group.add_argument(
        "--skip-ulimit",
        action="store_true",
        help="Skip attempting to set RLIMIT_MEMLOCK to unlimited before execution.",
    )
    runtime_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch command and environment changes without executing it.",
    )

    return parser.parse_args(normalize_extra_arg_options(argv))


def build_launch_command(args: argparse.Namespace) -> list[str]:
    """Build the SGLang launch_server command as a subprocess-ready list."""
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--served-model-name",
        args.served_model_name,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--tool-call-parser",
        args.tool_call_parser,
        "--reasoning-parser",
        args.reasoning_parser,
        "--speculative-algorithm",
        args.speculative_algorithm,
        "--speculative-num-steps",
        str(args.speculative_num_steps),
        "--speculative-eagle-topk",
        str(args.speculative_eagle_topk),
        "--speculative-num-draft-tokens",
        str(args.speculative_num_draft_tokens),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--nnodes",
        str(args.nnodes),
        "--node-rank",
        str(args.node_rank),
        "--dist-init-addr",
        args.dist_init_addr,
        "--disaggregation-mode",
        args.disaggregation_mode,
        "--disaggregation-transfer-backend",
        args.disaggregation_transfer_backend,
        "--disaggregation-ib-device",
        args.disaggregation_ib_device,
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    cmd.extend(
        [
            "--max-running-requests",
            str(args.max_running_requests),
            "--chunked-prefill-size",
            str(args.chunked_prefill_size),
            "--max-prefill-tokens",
            str(args.max_prefill_tokens),
        ]
    )
    cmd.extend(args.sglang_arg)
    return cmd


def build_runtime_env(
    args: argparse.Namespace,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment with proxies removed and NCCL defaults applied."""
    env = dict(os.environ if base_env is None else base_env)
    for proxy_key in PROXY_ENV_VARS:
        env.pop(proxy_key, None)

    env.update(
        {
            "NCCL_IB_GID_INDEX": str(args.nccl_ib_gid_index),
            "NCCL_IB_HCA": str(args.nccl_ib_hca),
            "NCCL_SOCKET_IFNAME": str(args.nccl_socket_ifname),
            "NCCL_IB_TC": str(args.nccl_ib_tc),
            "NCCL_IB_TIMEOUT": str(args.nccl_ib_timeout),
            "NCCL_IB_RETRY_CNT": str(args.nccl_ib_retry_cnt),
        }
    )
    return env


def apply_memlock_ulimit(skip_ulimit: bool) -> None:
    """Try to mirror `ulimit -l unlimited` on Unix before launching SGLang."""
    if skip_ulimit:
        return

    try:
        import resource
    except ImportError as exc:
        raise RuntimeError(
            "resource module is not available on this platform; rerun with --skip-ulimit "
            "or execute on Unix/Linux where RLIMIT_MEMLOCK is supported."
        ) from exc

    try:
        hard_limit = resource.getrlimit(resource.RLIMIT_MEMLOCK)[1]
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard_limit, hard_limit))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "failed to set RLIMIT_MEMLOCK to unlimited; run with sufficient privileges "
            "or pass --skip-ulimit if the limit is managed externally."
        ) from exc


def print_dry_run(cmd: Sequence[str], env: Mapping[str, str]) -> None:
    """Print the command and relevant environment changes for review."""
    print("Proxy environment removed:")
    for proxy_key in PROXY_ENV_VARS:
        print(f"  unset {proxy_key}")

    print("NCCL environment:")
    for env_key in NCCL_ENV_DEFAULTS:
        print(f"  {env_key}={env[env_key]}")

    print("Command:")
    print(shlex.join(cmd))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    cmd = build_launch_command(args)
    env = build_runtime_env(args)

    print_dry_run(cmd, env)
    if args.dry_run:
        return 0

    try:
        apply_memlock_ulimit(args.skip_ulimit)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    completed = subprocess.run(cmd, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
