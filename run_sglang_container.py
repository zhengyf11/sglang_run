#!/usr/bin/env python3
"""Run a long-lived SGLang Docker container without starting SGLang."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from typing import Sequence


DEFAULT_IMAGE = "lmsysorg/sglang:latest"
DEFAULT_CONTAINER_NAME = "sglang"
DEFAULT_SHM_SIZE = "32g"
KEEPALIVE_COMMAND = "tail -f /dev/null"


def normalize_extra_arg_options(argv: Sequence[str] | None) -> list[str] | None:
    """Allow --docker-arg values that start with a dash.

    argparse treats a value such as ``--ipc=host`` after ``--docker-arg`` as a
    new option instead of as that option's value. Converting only this known
    passthrough option to ``--key=value`` preserves the user-facing ``--key
    value`` form while still allowing raw Docker flags.
    """
    raw_argv = sys.argv[1:] if argv is None else list(argv)

    passthrough_options = {"--docker-arg"}
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
    """Parse command line arguments for the Docker keepalive runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a long-lived SGLang Docker container with /bin/bash keepalive; "
            "the script does not start an SGLang server process."
        )
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

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the docker command without executing it.",
    )
    return parser.parse_args(normalize_extra_arg_options(argv))


def build_docker_command(args: argparse.Namespace) -> list[str]:
    """Build the docker run command as a list suitable for subprocess.run."""
    cmd = ["docker", "run"]
    if args.rm:
        cmd.append("--rm")
    cmd.append("-d")
    if args.container_name:
        cmd.extend(["--name", args.container_name])
    if args.gpus and args.gpus.lower() != "none":
        cmd.extend(["--gpus", args.gpus])
    if args.shm_size:
        cmd.extend(["--shm-size", args.shm_size])

    cmd.extend(["--net=host", "--entrypoint", "/bin/bash"])

    for volume in args.volume:
        cmd.extend(["-v", volume])
    for env_var in args.env:
        cmd.extend(["-e", env_var])
    cmd.extend(args.docker_arg)

    cmd.extend([args.image, "-lc", KEEPALIVE_COMMAND])
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
