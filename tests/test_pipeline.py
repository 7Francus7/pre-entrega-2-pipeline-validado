import json
import os
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI, AuthenticationError, LengthFinishReasonError
from pydantic import ValidationError
from tenacity import wait_none

from chain import process_text
from schemas import EntidadesTecnicas, NivelCriticidad

VALID_OUTPUT = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "Servicio caído por agotamiento del pool de conexiones.",
}


def completion(content: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "chat-test", "object": "chat.completion", "created": 1,
        "model": "gpt-4.1-mini",
        "choices": [{
            "index": 0, "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": content},
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
    }


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "gpt-4.1-mini", "LLM_MAX_RETRIES": "0",
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.dotenv = patch("chain.load_dotenv")
        self.dotenv.start()
        self.addCleanup(self.dotenv.stop)
        self.wait = patch("langchain_core.runnables.retry.wait_exponential_jitter", return_value=wait_none())
        self.wait.start()
        self.addCleanup(self.wait.stop)

    async def run_pipeline(self, responses, text="FastAPI con Redis y PostgreSQL; servicio caído."):
        requests = []
        self.requests = requests

        def handler(request):
            requests.append(json.loads(request.content))
            return responses[len(requests) - 1]

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(http_client.aclose)
        sdk = AsyncOpenAI(api_key="test-key", http_client=http_client, max_retries=0)
        try:
            with patch("chain.AsyncOpenAI", return_value=sdk):
                result = await process_text(text)
        finally:
            self.assertTrue(http_client.is_closed)
        return result, requests

    async def test_valid_output_and_modular_prompt(self):
        text = "FastAPI falla con {datos} y Redis."
        result, requests = await self.run_pipeline([
            httpx.Response(200, json=completion(json.dumps(VALID_OUTPUT))),
        ], text)
        self.assertIsInstance(result, EntidadesTecnicas)
        self.assertEqual(result.nivel_de_criticidad, NivelCriticidad.alta)
        self.assertEqual(result.model_dump(mode="json"), VALID_OUTPUT)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["response_format"]["type"], "json_schema")
        self.assertIn(text, requests[0]["messages"][1]["content"])
        self.assertIn("tecnologias", requests[0]["messages"][0]["content"])

    async def test_retry_invalid_json_and_incomplete_fields(self):
        for content in ['', '{"tecnologias": [', '{"tecnologias": ["Redis"]}']:
            with self.subTest(content=content), self.assertLogs("chain", level="INFO") as logs:
                result, requests = await self.run_pipeline([
                    httpx.Response(200, json=completion(content)),
                    httpx.Response(200, json=completion(json.dumps(VALID_OUTPUT))),
                ])
                self.assertEqual(result.tecnologias, VALID_OUTPUT["tecnologias"])
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0]["messages"], requests[1]["messages"])
                self.assertTrue(any("Respuesta inválida" in line for line in logs.output))
                self.assertTrue(any("Validación correcta" in line for line in logs.output))

    async def test_retry_truncation_before_parsing(self):
        with self.assertLogs("chain", level="WARNING") as logs:
            result, requests = await self.run_pipeline([
                httpx.Response(200, json=completion("esto no es JSON", "length")),
                httpx.Response(200, json=completion(json.dumps(VALID_OUTPUT))),
            ])
        self.assertEqual(result.nivel_de_criticidad, NivelCriticidad.alta)
        self.assertEqual(len(requests), 2)
        self.assertTrue(any("LengthFinishReasonError" in line for line in logs.output))

    async def test_two_invalid_attempts_propagate_validation_error(self):
        with self.assertLogs("chain", level="ERROR") as logs:
            with self.assertRaises(ValidationError):
                await self.run_pipeline([
                    httpx.Response(200, json=completion("{}")),
                    httpx.Response(200, json=completion("{}")),
                ])
        self.assertTrue(any("después de dos intentos" in line for line in logs.output))
        self.assertEqual(len(self.requests), 2)

    async def test_two_truncated_attempts_propagate_length_error(self):
        with self.assertLogs("chain", level="ERROR"), self.assertRaises(LengthFinishReasonError):
            await self.run_pipeline([
                httpx.Response(200, json=completion("{", "length")),
                httpx.Response(200, json=completion("{", "length")),
            ])
        self.assertEqual(len(self.requests), 2)

    async def test_ambiguous_text_without_invented_technologies(self):
        data = {
            "tecnologias": [], "nivel_de_criticidad": "media",
            "resumen_tecnico": "Lentitud intermitente sin causa ni impacto confirmados.",
        }
        result, requests = await self.run_pipeline([
            httpx.Response(200, json=completion(json.dumps(data))),
        ], "A veces anda lento; no sé qué usa ni si afecta a alguien.")
        self.assertEqual(result.tecnologias, [])
        self.assertEqual(result.nivel_de_criticidad, NivelCriticidad.media)
        self.assertEqual(len(requests), 1)

    async def test_authentication_is_not_retried(self):
        with self.assertRaises(AuthenticationError):
            await self.run_pipeline([
                httpx.Response(401, json={"error": {"message": "Invalid API key", "type": "invalid_request_error"}}),
            ])
        self.assertEqual(len(self.requests), 1)

    async def test_empty_input_is_rejected_before_creating_client(self):
        for text in ["", "   ", None, 123]:
            with self.subTest(text=text), patch("chain.AsyncOpenAI") as client:
                with self.assertRaises(ValueError):
                    await process_text(text)
                client.assert_not_called()


class EntitySchemaTests(unittest.TestCase):
    def test_invalid_fields(self):
        for changes in [
            {"tecnologias": [""]}, {"tecnologias": ["   "]},
            {"tecnologias": [123]}, {"tecnologias": "Redis"},
            {"nivel_de_criticidad": "urgente"}, {"resumen_tecnico": "   "},
            {"campo_extra": "no permitido"},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                EntidadesTecnicas.model_validate(VALID_OUTPUT | changes)
