from __future__ import annotations

import subprocess
import unittest
from unittest import mock

import run_sglang_container


def option_values(command: list[str], option: str) -> list[str]:
    return [command[index + 1] for index, item in enumerate(command[:-1]) if item == option]


def assert_option_block(testcase: unittest.TestCase, command: list[str], option: str, expected_values: list[str]) -> None:
    option_indexes = [index for index, item in enumerate(command) if item == option]
    testcase.assertEqual(option_indexes, list(range(option_indexes[0], option_indexes[0] + len(option_indexes) * 2, 2)))
    testcase.assertEqual([command[index + 1] for index in option_indexes], expected_values)


class ParseArgsTests(unittest.TestCase):
    def test_model_is_not_required(self) -> None:
        args = run_sglang_container.parse_args([])

        self.assertEqual(args.image, "lmsysorg/sglang:latest")
        self.assertEqual(args.container_name, "sglang")
        self.assertEqual(args.volume, ["/sys/fs/cgroup:/sys/fs/cgroup:ro"])
        self.assertEqual(args.env, ["NVIDIA_VISIBLE_DEVICES=all"])

    def test_docker_passthrough_option_accepts_dash_prefixed_values(self) -> None:
        args = run_sglang_container.parse_args(
            [
                "--docker-arg",
                "--ipc=host",
            ]
        )

        self.assertEqual(args.docker_arg, ["--ipc=host"])

    def test_removed_sglang_and_port_options_are_rejected(self) -> None:
        removed_options = [
            ["--model", "model-id"],
            ["--host", "0.0.0.0"],
            ["--port", "30000"],
            ["--host-port", "30000"],
            ["--container-port", "30000"],
            ["--tp", "2"],
            ["--served-model-name", "qwen"],
            ["--mem-fraction-static", "0.8"],
            ["--sglang-arg", "--trust-remote-code"],
            ["--detach"],
            ["--gpus", "none"],
        ]

        for option in removed_options:
            with self.subTest(option=option), self.assertRaises(SystemExit):
                run_sglang_container.parse_args(option)


class BuildDockerCommandTests(unittest.TestCase):
    def test_builds_default_keepalive_command(self) -> None:
        args = run_sglang_container.parse_args([])

        cmd = run_sglang_container.build_docker_command(args)

        self.assertEqual(cmd[:4], ["docker", "run", "--rm", "-d"])
        self.assertIn("--name", cmd)
        self.assertEqual(cmd[cmd.index("--name") + 1], "sglang")
        self.assertIn("--shm-size", cmd)
        self.assertEqual(cmd[cmd.index("--shm-size") + 1], "32g")
        self.assertIn("--user=0", cmd)
        self.assertIn("--privileged", cmd)
        self.assertIn("--ipc=host", cmd)
        self.assertIn("--network", cmd)
        self.assertEqual(cmd[cmd.index("--network") + 1], "host")
        self.assertIn("--runtime=nvidia", cmd)
        self.assertIn("--gpus", cmd)
        self.assertEqual(cmd[cmd.index("--gpus") + 1], "all")
        self.assertIn("--ulimit", cmd)
        self.assertEqual(cmd[cmd.index("--ulimit") + 1], "memlock=-1:-1")
        self.assertIn("-v", cmd)
        self.assertIn("/sys/fs/cgroup:/sys/fs/cgroup:ro", cmd)
        self.assertIn("-e", cmd)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=all", cmd)
        assert_option_block(self, cmd, "-v", ["/sys/fs/cgroup:/sys/fs/cgroup:ro"])
        assert_option_block(self, cmd, "-e", ["NVIDIA_VISIBLE_DEVICES=all"])
        self.assertLess(cmd.index("NVIDIA_VISIBLE_DEVICES=all"), cmd.index("--entrypoint"))
        self.assertNotIn("--net=host", cmd)
        self.assertIn("--entrypoint", cmd)
        self.assertEqual(cmd[cmd.index("--entrypoint") + 1], "/bin/bash")
        self.assertIn("lmsysorg/sglang:latest", cmd)
        self.assertEqual(cmd[-3:], ["lmsysorg/sglang:latest", "-lc", "tail -f /dev/null"])
        self.assertNotIn("-p", cmd)
        self.assertNotIn("python3", cmd)
        self.assertNotIn("sglang.launch_server", cmd)
        self.assertNotIn("--model-path", cmd)
        self.assertNotIn("--port", cmd)

    def test_dynamic_container_options_are_applied(self) -> None:
        args = run_sglang_container.parse_args(
            [
                "--image",
                "custom/sglang:test",
                "--container-name",
                "custom-name",
                "--shm-size",
                "16g",
                "--volume",
                "/data/models:/models:ro",
                "--volume",
                "/data/cache:/cache",
                "--env",
                "HF_HOME=/cache",
                "--env",
                "TOKEN=value with spaces",
                "--docker-arg",
                "--ipc=host",
            ]
        )

        cmd = run_sglang_container.build_docker_command(args)

        self.assertIn("--gpus", cmd)
        self.assertEqual(cmd[cmd.index("--gpus") + 1], "all")
        self.assertEqual(cmd[cmd.index("--name") + 1], "custom-name")
        self.assertEqual(cmd[cmd.index("--shm-size") + 1], "16g")
        self.assertIn("--user=0", cmd)
        self.assertIn("--privileged", cmd)
        self.assertIn("--ipc=host", cmd)
        self.assertEqual(cmd[cmd.index("--network") + 1], "host")
        self.assertIn("--runtime=nvidia", cmd)
        self.assertEqual(cmd[cmd.index("--ulimit") + 1], "memlock=-1:-1")
        self.assertIn("/sys/fs/cgroup:/sys/fs/cgroup:ro", cmd)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=all", cmd)
        self.assertEqual(cmd[cmd.index("--entrypoint") + 1], "/bin/bash")
        self.assertIn("-v", cmd)
        self.assertIn("/data/models:/models:ro", cmd)
        self.assertIn("/data/cache:/cache", cmd)
        self.assertIn("-e", cmd)
        self.assertIn("HF_HOME=/cache", cmd)
        self.assertIn("TOKEN=value with spaces", cmd)
        assert_option_block(
            self,
            cmd,
            "-v",
            ["/sys/fs/cgroup:/sys/fs/cgroup:ro", "/data/models:/models:ro", "/data/cache:/cache"],
        )
        assert_option_block(self, cmd, "-e", ["NVIDIA_VISIBLE_DEVICES=all", "HF_HOME=/cache", "TOKEN=value with spaces"])
        self.assertLess(cmd.index("TOKEN=value with spaces"), cmd.index("--ipc=host", cmd.index("TOKEN=value with spaces")))
        self.assertLess(cmd.index("--ipc=host"), cmd.index("custom/sglang:test"))
        self.assertEqual(cmd[-3:], ["custom/sglang:test", "-lc", "tail -f /dev/null"])
        self.assertNotIn("-p", cmd)
        self.assertNotIn("sglang.launch_server", cmd)

    def test_no_rm_keeps_default_detached_mode(self) -> None:
        args = run_sglang_container.parse_args(["--no-rm"])

        cmd = run_sglang_container.build_docker_command(args)

        self.assertNotIn("--rm", cmd)
        self.assertIn("-d", cmd)


class MainTests(unittest.TestCase):
    def test_dry_run_does_not_call_subprocess(self) -> None:
        with mock.patch.object(run_sglang_container.subprocess, "run") as run_mock:
            exit_code = run_sglang_container.main(["--dry-run"])

        self.assertEqual(exit_code, 0)
        run_mock.assert_not_called()

    def test_main_executes_subprocess_with_list_command(self) -> None:
        completed = subprocess.CompletedProcess(args=["docker"], returncode=7)
        with mock.patch.object(
            run_sglang_container.subprocess,
            "run",
            return_value=completed,
        ) as run_mock:
            exit_code = run_sglang_container.main([])

        self.assertEqual(exit_code, 7)
        run_mock.assert_called_once()
        called_cmd = run_mock.call_args.args[0]
        self.assertIsInstance(called_cmd, list)
        self.assertEqual(called_cmd[0:2], ["docker", "run"])
        self.assertEqual(run_mock.call_args.kwargs, {"check": False})


if __name__ == "__main__":
    unittest.main()
