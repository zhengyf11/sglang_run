from __future__ import annotations

import io
import json
import unittest
from unittest import mock

import prefill_command_web


class CommandGenerationTests(unittest.TestCase):
    def test_builds_default_issue_command(self) -> None:
        response = prefill_command_web.build_command_response({})
        cmd = response["command"]

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
        self.assertFalse(response["executed"])
        self.assertIn("python3 -m sglang.launch_server", response["shell_command"])

    def test_blank_missing_and_none_fields_use_defaults(self) -> None:
        config = prefill_command_web.normalize_form_payload(
            {"model_path": "", "served_model_name": None, "tensor_parallel_size": "  "}
        )

        self.assertEqual(config["model_path"], "/mnt/GLM-5.1-FP8")
        self.assertEqual(config["served_model_name"], "GLM-5.1-FP8")
        self.assertEqual(config["tensor_parallel_size"], 8)
        self.assertEqual(config["port"], 30000)

    def test_dynamic_parameters_override_defaults(self) -> None:
        response = prefill_command_web.build_command_response(
            {
                "model_path": "/models/custom",
                "served_model_name": "custom-model",
                "tensor_parallel_size": 2,
                "host": "127.0.0.1",
                "port": 31000,
                "nnodes": 2,
                "node_rank": 1,
                "dist_init_addr": "10.0.0.1:20000",
                "disaggregation_transfer_backend": "custom-backend",
                "disaggregation_ib_device": "mlx5_0",
                "mem_fraction_static": 0.55,
                "max_running_requests": 64,
                "chunked_prefill_size": 4096,
                "max_prefill_tokens": 32768,
            }
        )
        cmd = response["command"]

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

    def test_boolean_options_control_flags(self) -> None:
        cmd = prefill_command_web.build_command_response(
            {"trust_remote_code": False, "disable_cuda_graph": False}
        )["command"]

        self.assertNotIn("--trust-remote-code", cmd)
        self.assertNotIn("--disable-cuda-graph", cmd)

    def test_extra_sglang_args_are_appended(self) -> None:
        cmd = prefill_command_web.build_command_response(
            {"extra_sglang_args": "--log-level debug --foo 'bar baz'"}
        )["command"]

        self.assertEqual(cmd[-4:], ["--log-level", "debug", "--foo", "bar baz"])

    def test_invalid_extra_sglang_args_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            prefill_command_web.build_command_response({"extra_sglang_args": "--unterminated '"})

    def test_env_exports_and_proxy_unsets_are_display_only(self) -> None:
        response = prefill_command_web.build_command_response(
            {"NCCL_IB_GID_INDEX": "5", "NCCL_IB_HCA": "mlx5_0"}
        )

        self.assertIn("export NCCL_IB_GID_INDEX=5", response["env_exports"])
        self.assertIn("export NCCL_IB_HCA=mlx5_0", response["env_exports"])
        self.assertIn("unset http_proxy", response["proxy_unsets"])
        self.assertIn("unset HTTPS_PROXY", response["proxy_unsets"])


class HandlerTests(unittest.TestCase):
    def _make_handler(self, method: str, path: str, body: bytes = b"") -> prefill_command_web.PrefillCommandHandler:
        handler = prefill_command_web.PrefillCommandHandler.__new__(prefill_command_web.PrefillCommandHandler)
        handler.path = path
        handler.command = method
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"{method} {path} HTTP/1.1"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        return handler

    def _json_response(self, handler: prefill_command_web.PrefillCommandHandler) -> dict[str, object]:
        raw = handler.wfile.getvalue().split(b"\r\n\r\n", 1)[1]
        return json.loads(raw.decode("utf-8"))

    def test_get_root_returns_form_and_output_window(self) -> None:
        html = prefill_command_web.render_index_html()

        self.assertIn("<form id=\"command-form\">", html)
        self.assertIn("Generated prefill shell command", html)
        self.assertIn("textarea id=\"command-output\"", html)
        self.assertIn("/api/command", html)

    def test_get_defaults_returns_default_config(self) -> None:
        handler = self._make_handler("GET", "/api/defaults")

        handler.do_GET()
        body = self._json_response(handler)

        self.assertEqual(body["defaults"]["model_path"], "/mnt/GLM-5.1-FP8")
        self.assertEqual(body["defaults"]["NCCL_IB_GID_INDEX"], "3")

    def test_post_api_command_returns_shell_command(self) -> None:
        handler = self._make_handler(
            "POST",
            "/api/command",
            json.dumps({"model_path": "/models/custom"}).encode("utf-8"),
        )

        handler.do_POST()
        body = self._json_response(handler)

        self.assertIn("shell_command", body)
        self.assertIn("--model-path /models/custom", body["shell_command"])
        self.assertFalse(body["executed"])

    def test_invalid_json_returns_400(self) -> None:
        handler = self._make_handler("POST", "/api/command", b"{")

        handler.do_POST()
        response = handler.wfile.getvalue().decode("utf-8")

        self.assertIn("400", response.splitlines()[0])
        self.assertIn("invalid JSON", response)

    def test_invalid_extra_args_returns_400(self) -> None:
        handler = self._make_handler(
            "POST",
            "/api/command",
            json.dumps({"extra_sglang_args": "--unterminated '"}).encode("utf-8"),
        )

        handler.do_POST()
        response = handler.wfile.getvalue().decode("utf-8")

        self.assertIn("400", response.splitlines()[0])
        self.assertIn("No closing quotation", response)


class MainSafetyTests(unittest.TestCase):
    def test_module_does_not_import_subprocess_or_expose_execution_endpoint(self) -> None:
        with open(prefill_command_web.__file__, encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertNotIn("import subprocess", source)
        self.assertNotIn("subprocess.run", source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("/start", source)
        self.assertNotIn("/run", source)
        self.assertNotIn("execute=true", source)

    def test_main_serves_web_ui_without_running_prefill(self) -> None:
        with mock.patch.object(prefill_command_web, "run_server") as run_server_mock:
            exit_code = prefill_command_web.main(["--host", "127.0.0.1", "--port", "9090"])

        self.assertEqual(exit_code, 0)
        run_server_mock.assert_called_once_with("127.0.0.1", 9090)


if __name__ == "__main__":
    unittest.main()
