from __future__ import annotations

import subprocess
import unittest
from unittest import mock

import run_sglang_container


class ParseArgsTests(unittest.TestCase):
    def test_model_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            run_sglang_container.parse_args([])

    def test_passthrough_options_accept_dash_prefixed_values(self) -> None:
        args = run_sglang_container.parse_args(
            [
                "--model",
                "model-id",
                "--docker-arg",
                "--ipc=host",
                "--sglang-arg",
                "--trust-remote-code",
            ]
        )

        self.assertEqual(args.docker_arg, ["--ipc=host"])
        self.assertEqual(args.sglang_arg, ["--trust-remote-code"])


class BuildDockerCommandTests(unittest.TestCase):
    def test_builds_default_command_with_required_model(self) -> None:
        args = run_sglang_container.parse_args(["--model", "Qwen/Qwen2.5-7B-Instruct"])

        cmd = run_sglang_container.build_docker_command(args)

        self.assertEqual(cmd[:3], ["docker", "run", "--rm"])
        self.assertIn("--name", cmd)
        self.assertEqual(cmd[cmd.index("--name") + 1], "sglang")
        self.assertIn("--gpus", cmd)
        self.assertEqual(cmd[cmd.index("--gpus") + 1], "all")
        self.assertIn("--shm-size", cmd)
        self.assertEqual(cmd[cmd.index("--shm-size") + 1], "32g")
        self.assertIn("-p", cmd)
        self.assertEqual(cmd[cmd.index("-p") + 1], "30000:30000")
        self.assertIn("lmsysorg/sglang:latest", cmd)
        self.assertIn("python3", cmd)
        self.assertIn("sglang.launch_server", cmd)
        self.assertIn("--model-path", cmd)
        self.assertEqual(cmd[cmd.index("--model-path") + 1], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(cmd[cmd.index("--host") + 1], "0.0.0.0")
        self.assertEqual(cmd[cmd.index("--port") + 1], "30000")
        self.assertEqual(cmd[cmd.index("--tp") + 1], "1")

    def test_dynamic_docker_and_sglang_options_are_applied(self) -> None:
        args = run_sglang_container.parse_args(
            [
                "--image",
                "custom/sglang:test",
                "--container-name",
                "custom-name",
                "--model",
                "/models/qwen",
                "--host-port",
                "18000",
                "--container-port",
                "30000",
                "--gpus",
                "none",
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
                "--host",
                "127.0.0.1",
                "--served-model-name",
                "qwen",
                "--tp",
                "2",
                "--mem-fraction-static",
                "0.8",
                "--sglang-arg",
                "--trust-remote-code",
            ]
        )

        cmd = run_sglang_container.build_docker_command(args)

        self.assertNotIn("--gpus", cmd)
        self.assertEqual(cmd[cmd.index("--name") + 1], "custom-name")
        self.assertEqual(cmd[cmd.index("--shm-size") + 1], "16g")
        self.assertEqual(cmd[cmd.index("-p") + 1], "18000:30000")
        self.assertIn("-v", cmd)
        self.assertIn("/data/models:/models:ro", cmd)
        self.assertIn("/data/cache:/cache", cmd)
        self.assertIn("-e", cmd)
        self.assertIn("HF_HOME=/cache", cmd)
        self.assertIn("TOKEN=value with spaces", cmd)
        self.assertLess(cmd.index("--ipc=host"), cmd.index("custom/sglang:test"))
        self.assertEqual(cmd[cmd.index("--model-path") + 1], "/models/qwen")
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
        self.assertEqual(cmd[cmd.index("--port") + 1], "30000")
        self.assertEqual(cmd[cmd.index("--tp") + 1], "2")
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "qwen")
        self.assertEqual(cmd[cmd.index("--mem-fraction-static") + 1], "0.8")
        self.assertIn("--trust-remote-code", cmd)

    def test_detach_and_no_rm_options(self) -> None:
        args = run_sglang_container.parse_args(
            ["--model", "model-id", "--detach", "--no-rm"]
        )

        cmd = run_sglang_container.build_docker_command(args)

        self.assertNotIn("--rm", cmd)
        self.assertIn("-d", cmd)


class MainTests(unittest.TestCase):
    def test_dry_run_does_not_call_subprocess(self) -> None:
        with mock.patch.object(run_sglang_container.subprocess, "run") as run_mock:
            exit_code = run_sglang_container.main(["--model", "model-id", "--dry-run"])

        self.assertEqual(exit_code, 0)
        run_mock.assert_not_called()

    def test_main_executes_subprocess_with_list_command(self) -> None:
        completed = subprocess.CompletedProcess(args=["docker"], returncode=7)
        with mock.patch.object(
            run_sglang_container.subprocess,
            "run",
            return_value=completed,
        ) as run_mock:
            exit_code = run_sglang_container.main(["--model", "model-id"])

        self.assertEqual(exit_code, 7)
        run_mock.assert_called_once()
        called_cmd = run_mock.call_args.args[0]
        self.assertIsInstance(called_cmd, list)
        self.assertEqual(called_cmd[0:2], ["docker", "run"])
        self.assertEqual(run_mock.call_args.kwargs, {"check": False})


if __name__ == "__main__":
    unittest.main()
