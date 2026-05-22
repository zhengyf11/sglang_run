#!/usr/bin/env python3
"""Run an SGLang server in a Docker container."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from typing import Sequence


DEFAULT_IMAGE = "lmsysorg/sglang:latest"
DEFAULT_CONTAINER_NAME = "sglang"
DEFAULT_PORT = "30000"
DEFAULT_SHM_SIZE = "32g"


def normalize_extra_arg_options(argv: Sequence[str] | None) -> list[str] | None:
    """Allow --docker-arg/--sglang-arg values that start with a dash.

    argparse treats a value such as ``--ipc=host`` after ``--docker-arg`` as a
    new option instead of as that option's value. Converting only these known
    passthrough options to ``--key=value`` preserves the user-facing ``--key
    value`` form while still allowing raw Docker/SGLang flags.
    """
    raw_argv = sys.argv[1:] if argv is None else list(argv)

    passthrough_options = {"--docker-arg", "--sglang-arg"}
    normalized: list[str] = []
    index = 0
    while index < len(raw_argv):
        item = raw_argv[index]
        if item in passthrough_options and index + 1 < len(raw_argv):
            normalized.append(f"{item}={raw_argv[index + 1]}")
            index += 2
            continue
        normalized.append(item)
        index += 1
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the SGLang Docker runner."""
    parser = argparse.ArgumentParser(
        description="Run an SGLang Docker container with configurable options."
    )

    docker_group = parser.add_argument_group("Docker container options")
    docker_group.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Docker image name. Defaults to {DEFAULT_IMAGE}.",
    )
    docker_group.add_argument(
        "--container-name",
        default=DEFAULT_CONTAINER_NAME,
        help=f"Docker container name. Defaults to {DEFAULT_CONTAINER_NAME}.",
    )
    docker_group.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=(
            "Default service port used for both host and container when "
            "--host-port or --container-port is not provided."
        ),
    )
    docker_group.add_argument(
        "--host-port",
        help="Host port to expose. Defaults to --port.",
    )
    docker_group.add_argument(
        "--container-port",
        help="Container port exposed by SGLang. Defaults to --port.",
    )
    docker_group.add_argument(
        "--gpus",
        default="all",
        help="Docker --gpus value, for example: all, device=0, or none. Defaults to all.",
    )
    docker_group.add_argument(
        "--shm-size",
        default=DEFAULT_SHM_SIZE,
        help=f"Docker --shm-size value. Defaults to {DEFAULT_SHM_SIZE}.",
    )
    docker_group.add_argument(
        "--detach",
        action="store_true",
        help="Run the container in the background with docker run -d.",
    )
    docker_group.add_argument(
        "--rm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove the container when it exits. Use --no-rm to keep it. Defaults to true.",
    )
    docker_group.add_argument(
        "--volume",
        action="append",
        default=[],
        metavar="HOST:CONTAINER[:MODE]",
        help="Docker volume mapping. Can be repeated.",
    )
    docker_group.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable passed to Docker. Can be repeated.",
    )
    docker_group.add_argument(
        "--docker-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra raw Docker argument appended before the image. Can be repeated.",
    )

    sglang_group = parser.add_argument_group("SGLang server options")
    sglang_group.add_argument(
        "--model",
        required=True,
        help="Model path or model ID passed to SGLang as --model-path.",
    )
    sglang_group.add_argument(
        "--host",
        default="0.0.0.0",
        help="SGLang bind host inside the container. Defaults to 0.0.0.0.",
    )
    sglang_group.add_argument(
        "--served-model-name",
        help="Optional served model name passed to SGLang.",
    )
    sglang_group.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size passed as --tp. Defaults to 1.",
    )
    sglang_group.add_argument(
        "--mem-fraction-static",
        type=float,
        help="Optional SGLang --mem-fraction-static value.",
    )
    sglang_group.add_argument(
        "--sglang-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra raw SGLang launch_server argument appended after built-in options. Can be repeated.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the docker command without executing it.",
    )
    return parser.parse_args(normalize_extra_arg_options(argv))


def build_docker_command(args: argparse.Namespace) -> list[str]:
    """Build the docker run command as a list suitable for subprocess.run."""
    host_port = args.host_port or args.port
    container_port = args.container_port or args.port

    cmd = ["docker", "run"]
    if args.rm:
        cmd.append("--rm")
    if args.detach:
        cmd.append("-d")
    if args.container_name:
        cmd.extend(["--name", args.container_name])
    if args.gpus and args.gpus.lower() != "none":
        cmd.extend(["--gpus", args.gpus])
    if args.shm_size:
        cmd.extend(["--shm-size", args.shm_size])

    cmd.extend(["-p", f"{host_port}:{container_port}"])

    for volume in args.volume:
        cmd.extend(["-v", volume])
    for env_var in args.env:
        cmd.extend(["-e", env_var])
    cmd.extend(args.docker_arg)

    cmd.append(args.image)
    cmd.extend(
        [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model,
            "--host",
            args.host,
            "--port",
            str(container_port),
            "--tp",
            str(args.tp),
        ]
    )
    if args.served_model_name:
        cmd.extend(["--served-model-name", args.served_model_name])
    if args.mem_fraction_static is not None:
        cmd.extend(["--mem-fraction-static", str(args.mem_fraction_static)])
    cmd.extend(args.sglang_arg)

    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    cmd = build_docker_command(args)

    print("Command:")
    print(shlex.join(cmd))
    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
