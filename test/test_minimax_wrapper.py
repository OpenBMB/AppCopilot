import numpy as np
import pytest

from wrappers.minimax_wrapper import (
    MINIMAX_ENDPOINTS,
    MINIMAX_MODELS,
    MiniMaxLlmWrapper,
)


ASSISTANT_TEXT = '{"thought":"done","STATUS":"finish"}'


class FakeOpenAIResponse:
    def model_dump(self):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": ASSISTANT_TEXT}}
            ]
        }


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeOpenAIResponse()


class FakeOpenAIClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


class FakeHttpResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"content": [{"type": "text", "text": ASSISTANT_TEXT}]}


class FakeSession:
    def __init__(self):
        self.request = None

    def post(self, url, **kwargs):
        self.request = {"url": url, **kwargs}
        return FakeHttpResponse()


def test_minimax_registry_contains_current_models_and_regional_endpoints():
    assert tuple(MINIMAX_MODELS) == ("MiniMax-M3", "MiniMax-M2.7")
    assert MINIMAX_MODELS["MiniMax-M3"]["input_modalities"] == (
        "text",
        "image",
        "video",
    )
    assert MINIMAX_MODELS["MiniMax-M2.7"]["input_modalities"] == ("text",)
    assert MINIMAX_ENDPOINTS == {
        "global_en": {
            "openai": "https://api.minimax.io/v1",
            "anthropic": "https://api.minimax.io/anthropic",
        },
        "cn_zh": {
            "openai": "https://api.minimaxi.com/v1",
            "anthropic": "https://api.minimaxi.com/anthropic",
        },
    }


def test_openai_transport_preserves_image_input():
    client = FakeOpenAIClient()
    wrapper = MiniMaxLlmWrapper(
        api_key="test-key",
        model_name="MiniMax-M3",
        region="global_en",
        transport="openai",
        client=client,
    )

    result = wrapper.predict_mm(
        "Open settings", [np.zeros((2, 2, 3), dtype=np.uint8)]
    )

    request = client.chat.completions.kwargs
    assert request["model"] == "MiniMax-M3"
    image_block = request["messages"][-1]["content"][1]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert result[3]["STATUS"] == "finish"


def test_openai_transport_accepts_previous_step_result():
    client = FakeOpenAIClient()
    wrapper = MiniMaxLlmWrapper(
        api_key="test-key",
        model_name="MiniMax-M3",
        region="global_en",
        transport="openai",
        client=client,
    )

    wrapper.predict_mm(
        "Continue the task",
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        last_result={"account": "demo"},
    )

    text_block = client.chat.completions.kwargs["messages"][-1]["content"][0]
    assert "Previous step result: {'account': 'demo'}" in text_block["text"]


def test_anthropic_transport_uses_china_endpoint_and_image_format():
    session = FakeSession()
    wrapper = MiniMaxLlmWrapper(
        api_key="test-key",
        model_name="MiniMax-M3",
        region="cn_zh",
        transport="anthropic",
        session=session,
    )

    result = wrapper.predict_mm(
        "Open settings", [np.zeros((2, 2, 3), dtype=np.uint8)]
    )

    assert (
        session.request["url"]
        == "https://api.minimaxi.com/anthropic/v1/messages"
    )
    assert session.request["headers"]["Authorization"] == "Bearer test-key"
    payload = session.request["json"]
    assert payload["model"] == "MiniMax-M3"
    image_block = payload["messages"][-1]["content"][1]
    assert image_block["type"] == "image"
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/jpeg"
    assert result[3]["STATUS"] == "finish"


def test_text_only_model_accepts_text_and_rejects_images():
    client = FakeOpenAIClient()
    wrapper = MiniMaxLlmWrapper(
        api_key="test-key",
        model_name="MiniMax-M2.7",
        region="global_en",
        transport="openai",
        client=client,
    )

    wrapper.predict("Summarize the task")
    assert len(client.chat.completions.kwargs["messages"][-1]["content"]) == 1

    with pytest.raises(ValueError, match="does not support image input"):
        wrapper.predict_mm(
            "Open settings", [np.zeros((2, 2, 3), dtype=np.uint8)]
        )
