from __future__ import annotations

import subprocess
import unittest
from unittest import mock

import run_sglang_prefill


class ParseArgsTests(unittest.TestCase):
    def test_sglang_arg_accepts_dash_prefixed_values(self) -> None:
        args = run_sglang_prefill.parse_args(
            [
                "--sglang-arg",
                "--log-level",
                "--sglang-arg",
                "debug",
            ]
        )

        self.assertEqual(args.sglang_arg, ["--log-level", "debug"])

    def test_tp_alias_sets_tensor_parallel_size(self) -> None:
        args = run_sglang_prefill.parse_args(["--tp", "4"])

        self.assertEqual(args.tensor_parallel_size, 4)


class BuildLaunchCommandTests(unittest.TestCase):
    def test_builds_default_issue_command(self) -> None:
        args = run_sglang_prefill.parse_args([])

        cmd = run_sglang_prefill.build_launch_command(args)

        self.assertEqual(cmd[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertEqual(cmd[cmd.index("--model-path") + 1], "/mnt/GLM-5.1-FP8")
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "GLM-5.1-FP8")
        self.assertEqual(cmd[cmd.index("--tensor-parallel-size") + 1], "8")
        self.assertEqual(cmd[cmd.index("--tool-call-parser") + 1], "glm47")
        self.assertEqual(cmd[cmd.index("--reasoning-parser") + 1], "glm45")
        self.assertEqual(cmd[cmd.index("--speculative-algorithm") + 1], "EAGLE")
        self.assertEqual(cmd[cmd.index("--speculative-num-steps") + 1], "3")
        self.assertEqual(cmd[cmd.index("--speculative-eagle-topk") + 1], "1")
        self.assertEqual(cmd[cmd.index("--speculative-num-draft-tokens") + 1], "4")
        self.assertEqual(cmd[cmd.index("--mem-fraction-static") + 1], "0.7")
        self.assertEqual(cmd[cmd.index("--host") + 1], "0.0.0.0")
        self.assertEqual(cmd[cmd.index("--port") + 1], "30000")
        self.assertEqual(cmd[cmd.index("--nnodes") + 1], "1")
        self.assertEqual(cmd[cmd.index("--node-rank") + 1], "0")
        self.assertEqual(cmd[cmd.index("--dist-init-addr") + 1], "127.0.0.1:20000")
        self.assertEqual(cmd[cmd.index("--disaggregation-mode") + 1], "prefill")
        self.assertEqual(cmd[cmd.index("--disaggregation-transfer-backend") + 1], "mooncake")
        self.assertEqual(
            cmd[cmd.index("--disaggregation-ib-device") + 1],
            "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,"
            "mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7",
        )
        self.assertIn("--trust-remote-code", cmd)
        self.assertIn("--disable-cuda-graph", cmd)
        self.assertEqual(cmd[cmd.index("--max-running-requests") + 1], "128")
        self.assertEqual(cmd[cmd.index("--chunked-prefill-size") + 1], "8192")
        self.assertEqual(cmd[cmd.index("--max-prefill-tokens") + 1], "65536")

    def test_dynamic_parameters_override_defaults(self) -> None:
        args = run_sglang_prefill.parse_args(
            [
                "--model-path",
                "/models/custom",
                "--served-model-name",
                "custom-model",
                "--tensor-parallel-size",
                "2",
                "--host",
                "127.0.0.1",
                "--port",
                "31000",
                "--nnodes",
                "2",
                "--node-rank",
                "1",
                "--dist-init-addr",
                "10.0.0.1:20000",
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-transfer-backend",
                "custom-backend",
                "--disaggregation-ib-device",
                "mlx5_0",
                "--mem-fraction-static",
                "0.55",
                "--max-running-requests",
                "64",
                "--chunked-prefill-size",
                "4096",
                "--max-prefill-tokens",
                "32768",
                "--no-trust-remote-code",
                "--enable-cuda-graph",
            ]
        )

        cmd = run_sglang_prefill.build_launch_command(args)

        self.assertEqual(cmd[cmd.index("--model-path") + 1], "/models/custom")
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "custom-model")
        self.assertEqual(cmd[cmd.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
        self.assertEqual(cmd[cmd.index("--port") + 1], "31000")
        self.assertEqual(cmd[cmd.index("--nnodes") + 1], "2")
        self.assertEqual(cmd[cmd.index("--node-rank") + 1], "1")
        self.assertEqual(cmd[cmd.index("--dist-init-addr") + 1], "10.0.0.1:20000")
        self.assertEqual(cmd[cmd.index("--disaggregation-transfer-backend") + 1], "custom-backend")
        self.assertEqual(cmd[cmd.index("--disaggregation-ib-device") + 1], "mlx5_0")
        self.assertEqual(cmd[cmd.index("--mem-fraction-static") + 1], "0.55")
        self.assertEqual(cmd[cmd.index("--max-running-requests") + 1], "64")
        self.assertEqual(cmd[cmd.index("--chunked-prefill-size") + 1], "4096")
        self.assertEqual(cmd[cmd.index("--max-prefill-tokens") + 1], "32768")
        self.assertNotIn("--trust-remote-code", cmd)
        self.assertNotIn("--disable-cuda-graph", cmd)

    def test_sglang_arg_passthrough_appends_extra_arguments(self) -> None:
        args = run_sglang_prefill.parse_args(
            [
                "--sglang-arg",
                "--log-level",
                "--sglang-arg",
                "debug",
            ]
        )

        cmd = run_sglang_prefill.build_launch_command(args)

        self.assertEqual(cmd[-2:], ["--log-level", "debug"])


class BuildRuntimeEnvTests(unittest.TestCase):
    def test_removes_proxy_environment_and_sets_nccl_defaults(self) -> None:
        args = run_sglang_prefill.parse_args([])
        base_env = {
            "PATH": "/usr/bin",
            "http_proxy": "http://proxy",
            "HTTPS_PROXY": "https://proxy",
            "ALL_PROXY": "socks://proxy",
        }

        env = run_sglang_prefill.build_runtime_env(args, base_env)

        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertNotIn("http_proxy", env)
        self.assertNotIn("HTTPS_PROXY", env)
        self.assertNotIn("ALL_PROXY", env)
        self.assertEqual(env["NCCL_IB_GID_INDEX"], "3")
        self.assertEqual(env["NCCL_IB_HCA"], "^mlx5_bond")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "bond0")
        self.assertEqual(env["NCCL_IB_TC"], "128")
        self.assertEqual(env["NCCL_IB_TIMEOUT"], "22")
        self.assertEqual(env["NCCL_IB_RETRY_CNT"], "15")

    def test_nccl_environment_can_be_overridden(self) -> None:
        args = run_sglang_prefill.parse_args(
            [
                "--nccl-ib-gid-index",
                "5",
                "--nccl-ib-hca",
                "mlx5_0",
                "--nccl-socket-ifname",
                "eth0",
                "--nccl-ib-tc",
                "64",
                "--nccl-ib-timeout",
                "30",
                "--nccl-ib-retry-cnt",
                "7",
            ]
        )

        env = run_sglang_prefill.build_runtime_env(args, {})

        self.assertEqual(env["NCCL_IB_GID_INDEX"], "5")
        self.assertEqual(env["NCCL_IB_HCA"], "mlx5_0")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "eth0")
        self.assertEqual(env["NCCL_IB_TC"], "64")
        self.assertEqual(env["NCCL_IB_TIMEOUT"], "30")
        self.assertEqual(env["NCCL_IB_RETRY_CNT"], "7")


class MainTests(unittest.TestCase):
    def test_dry_run_does_not_apply_ulimit_or_call_subprocess(self) -> None:
        with mock.patch.object(run_sglang_prefill, "apply_memlock_ulimit") as ulimit_mock:
            with mock.patch.object(run_sglang_prefill.subprocess, "run") as run_mock:
                exit_code = run_sglang_prefill.main(["--dry-run"])

        self.assertEqual(exit_code, 0)
        ulimit_mock.assert_not_called()
        run_mock.assert_not_called()

    def test_main_executes_subprocess_with_list_env_and_check_false(self) -> None:
        completed = subprocess.CompletedProcess(args=["python3"], returncode=9)
        with mock.patch.object(run_sglang_prefill, "apply_memlock_ulimit") as ulimit_mock:
            with mock.patch.object(
                run_sglang_prefill.subprocess,
                "run",
                return_value=completed,
            ) as run_mock:
                exit_code = run_sglang_prefill.main(["--skip-ulimit"])

        self.assertEqual(exit_code, 9)
        ulimit_mock.assert_called_once_with(True)
        run_mock.assert_called_once()
        called_cmd = run_mock.call_args.args[0]
        called_kwargs = run_mock.call_args.kwargs
        self.assertIsInstance(called_cmd, list)
        self.assertEqual(called_cmd[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertIn("env", called_kwargs)
        self.assertIsInstance(called_kwargs["env"], dict)
        self.assertEqual(called_kwargs["check"], False)
        self.assertNotIn("shell", called_kwargs)

    def test_ulimit_error_returns_clear_failure(self) -> None:
        with mock.patch.object(
            run_sglang_prefill,
            "apply_memlock_ulimit",
            side_effect=RuntimeError("ulimit failed"),
        ):
            with mock.patch.object(run_sglang_prefill.subprocess, "run") as run_mock:
                exit_code = run_sglang_prefill.main([])

        self.assertEqual(exit_code, 1)
        run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
