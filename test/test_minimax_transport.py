"""Offline request-boundary coverage using the real pinned SDK and requests."""

import argparse
import ast
import base64
import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import requests
from PIL import Image

from wrappers import minimax_wrapper


ANSWER = '{"thought":"done","STATUS":"finish"}'
THINKING = {"type": "thinking", "thinking": "Inspect the screen.", "signature": "test-signature"}


@pytest.fixture
def captured_requests(monkeypatch):
    captured = []

    def openai_send(transport, request):
        captured.append((str(request.url), json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "test-response",
                "object": "chat.completion",
                "created": 0,
                "model": captured[-1][1]["model"],
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": ANSWER,
                        "reasoning_content": "Inspect the screen.",
                    },
                }],
            },
        )

    def anthropic_send(session, request, **kwargs):
        captured.append((request.url, json.loads(request.body)))
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({
            "content": [THINKING, {"type": "text", "text": ANSWER}],
        }).encode()
        return response

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", openai_send)
    monkeypatch.setattr(requests.Session, "send", anthropic_send)
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(minimax_wrapper.MiniMaxLlmWrapper, "RETRY_WAITING_SECONDS", 0)
    return captured


def assert_api_defaults(payload):
    for field in ("thinking", "reasoning_effort", "output_config", "service_tier", "reasoning_split"):
        assert field not in payload


def assert_screenshot(content, transport):
    block = content[1]
    if transport == "openai":
        prefix, encoded = block["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/jpeg;base64"
    else:
        assert block["source"]["media_type"] == "image/jpeg"
        encoded = block["source"]["data"]
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert image.format == "JPEG"
    assert image.size == (2, 2)


@pytest.mark.parametrize("model", [None, "MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-model"])
@pytest.mark.parametrize("region", ["global_en", "cn_zh"])
@pytest.mark.parametrize("transport", ["anthropic", "openai"])
def test_serialized_requests_preserve_selection_and_history(captured_requests, model, region, transport):
    options = {} if model is None else {"model_name": model}
    wrapper = minimax_wrapper.create_minimax_wrapper(
        region=region, transport=transport, use_history=True, max_retry=1, **options
    )
    expected_model = model or "MiniMax-M3.1-Flash"
    images = [np.zeros((2, 2, 3), dtype=np.uint8)] if expected_model in minimax_wrapper.MINIMAX_MODELS else []
    result = wrapper.predict_mm("Open settings", images, last_result={"account": "demo"})
    assert result[3]["STATUS"] == "finish"
    url, payload = captured_requests[0]
    endpoint = minimax_wrapper.MINIMAX_ENDPOINTS[region][transport]
    assert url == endpoint + ("/chat/completions" if transport == "openai" else "/v1/messages")
    assert payload["model"] == expected_model
    assert_api_defaults(payload)
    user_content = payload["messages"][-1]["content"]
    assert "Previous step result: {'account': 'demo'}" in user_content[0]["text"]
    if images:
        assert_screenshot(user_content, transport)
    else:
        assert len(user_content) == 1

    assert wrapper.predict("Continue")[3]["STATUS"] == "finish"
    followup = captured_requests[1][1]
    assert followup["model"] == expected_model
    assert_api_defaults(followup)
    assistant = followup["messages"][-2]
    assert assistant["role"] == "assistant"
    if transport == "openai":
        assert assistant["reasoning_content"] == "Inspect the screen."
        assert assistant["content"] == ANSWER
    else:
        assert assistant["content"] == [THINKING, {"type": "text", "text": ANSWER}]
    assert followup["messages"][-3]["content"] == user_content


def agent_selection_classes():
    """Execute the actual CLI and factory methods without OCR/device imports."""
    source = Path(__file__).resolve().parents[1] / "run_agent.py"
    tree = ast.parse(source.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ArgumentHandler":
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "GUITaskExecutor":
            node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in ("__init__", "_create_llm")]
            selected.append(node)
    namespace = dict(vars(minimax_wrapper), argparse=argparse, OCRService=object, AVAILABLE_TASKS={})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("model", [None, "MiniMax-M3", "MiniMax-M2.7", "custom-model"])
def test_cli_selection_reaches_default_http_transport(captured_requests, model):
    namespace = agent_selection_classes()
    handler = namespace["ArgumentHandler"]()
    argv = ["--custom-task", "--llm-provider", "minimax"]
    if model is not None:
        argv += ["--llm-model", model]
    args = handler.parser.parse_args(argv)
    handler.validate_args(args)
    help_text = handler.parser.format_help()
    assert "MiniMax-M3.1-Flash, MiniMax-M3" in help_text
    assert "MiniMax-M2.7" not in help_text
    executor = namespace["GUITaskExecutor"](
        ocr_service=None, llm_provider=args.llm_provider, llm_model=args.llm_model,
        llm_region=args.llm_region, llm_transport=args.llm_transport,
    )
    assert executor._create_llm().predict("Continue")[3]["STATUS"] == "finish"
    url, payload = captured_requests[0]
    assert url == "https://api.minimax.io/anthropic/v1/messages"
    assert payload["model"] == (model or "MiniMax-M3.1-Flash")


def test_local_provider_remains_default():
    namespace = agent_selection_classes()
    namespace["MiniCPMWrapper"] = lambda **kwargs: kwargs
    args = namespace["ArgumentHandler"]().parser.parse_args(["--custom-task"])
    assert args.llm_provider == "local"
    executor = namespace["GUITaskExecutor"](ocr_service=None)
    assert executor._create_llm()["model_name"] == "AgentCPM-GUI"


@pytest.mark.parametrize("model", ["", "   "])
def test_empty_explicit_model_is_rejected(model):
    with pytest.raises(ValueError, match="model ID is required"):
        minimax_wrapper.create_minimax_wrapper(api_key="test-key", model_name=model)
