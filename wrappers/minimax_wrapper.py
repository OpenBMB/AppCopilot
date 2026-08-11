import os
import time
from typing import Any

import numpy as np
import requests
from openai import OpenAI

from wrappers.constants import ERROR_CALLING_LLM
from wrappers.openai_wrapper import OpenAILlmWrapper


MINIMAX_MODELS = {
    "MiniMax-M3": {"input_modalities": ("text", "image", "video")},
    "MiniMax-M2.7": {"input_modalities": ("text",)},
}

MINIMAX_ENDPOINTS = {
    "global_en": {
        "openai": "https://api.minimax.io/v1",
        "anthropic": "https://api.minimax.io/anthropic",
    },
    "cn_zh": {
        "openai": "https://api.minimaxi.com/v1",
        "anthropic": "https://api.minimaxi.com/anthropic",
    },
}

MINIMAX_TRANSPORTS = ("anthropic", "openai")
DEFAULT_MINIMAX_MODEL = "MiniMax-M3"
DEFAULT_MINIMAX_REGION = "global_en"
DEFAULT_MINIMAX_TRANSPORT = "anthropic"


class MiniMaxLlmWrapper(OpenAILlmWrapper):
    """Multimodal MiniMax wrapper for compatible API transports."""

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MINIMAX_MODEL,
        region: str = DEFAULT_MINIMAX_REGION,
        transport: str = DEFAULT_MINIMAX_TRANSPORT,
        max_retry: int = 3,
        temperature: float = 0.1,
        use_history: bool = False,
        history_size: int = 10,
        max_tokens: int = 4096,
        request_timeout: int = 120,
        client: Any = None,
        session: Any = None,
    ):
        if model_name not in MINIMAX_MODELS:
            raise ValueError(f"Unsupported MiniMax model: {model_name}")
        if region not in MINIMAX_ENDPOINTS:
            raise ValueError(f"Unsupported MiniMax region: {region}")
        if transport not in MINIMAX_TRANSPORTS:
            raise ValueError(f"Unsupported MiniMax transport: {transport}")
        if not api_key:
            raise ValueError("MINIMAX_API_KEY is required for the MiniMax provider.")

        self.region = region
        self.transport = transport
        self.base_url = MINIMAX_ENDPOINTS[region][transport]
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.session = requests.Session() if session is None else session

        if transport == "openai" and client is None:
            client = OpenAI(api_key=api_key, base_url=self.base_url)

        super().__init__(
            model_name=model_name,
            max_retry=max_retry,
            temperature=temperature,
            use_history=use_history,
            history_size=history_size,
            client=client,
            supports_images=(
                "image" in MINIMAX_MODELS[model_name]["input_modalities"]
            ),
        )

    @staticmethod
    def _to_anthropic_content(content: Any) -> Any:
        if isinstance(content, str):
            return content

        converted = []
        for block in content:
            if block["type"] == "text":
                converted.append(block)
                continue
            if block["type"] != "image_url":
                raise ValueError(f"Unsupported content block: {block['type']}")

            data_url = block["image_url"]["url"]
            prefix, encoded = data_url.split(",", 1)
            media_type = prefix.removeprefix("data:").removesuffix(";base64")
            converted.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": encoded,
                    },
                }
            )
        return converted

    def _prepare_anthropic_request(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[dict, Any]:
        messages = self._prepare_messages(text_prompt, images)
        system_content = messages[0]["content"]
        system = "\n".join(
            block["text"] for block in system_content if block["type"] == "text"
        )
        conversation = [
            {
                "role": message["role"],
                "content": self._to_anthropic_content(message["content"]),
            }
            for message in messages[1:]
        ]
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": conversation,
        }
        return payload, messages[-1]["content"]

    def predict_mm(
        self,
        text_prompt: str,
        images: list[np.ndarray],
        last_result: Any = None,
    ) -> tuple[str, Any, Any, Any]:
        if last_result is not None:
            text_prompt += f"\nPrevious step result: {last_result}"

        if self.transport == "openai":
            return super().predict_mm(text_prompt, images)

        self._validate_images(images)
        payload, history_content = self._prepare_anthropic_request(
            text_prompt, images
        )
        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS

        while counter > 0:
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/messages",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                response_data = response.json()
                assistant_text = "".join(
                    block["text"]
                    for block in response_data["content"]
                    if block.get("type") == "text"
                )
                if not assistant_text:
                    raise ValueError("MiniMax response did not contain a text block.")

                action = self.extract_and_validate_json(assistant_text)
                self._push_history("user", history_content)
                self._push_history("assistant", assistant_text)
                return assistant_text, None, response_data, action
            except Exception as error:
                print(error)
                counter -= 1
                if counter > 0:
                    time.sleep(wait_seconds)
                    wait_seconds *= 2
                    print("Error calling LLM, will retry soon...")

        return ERROR_CALLING_LLM, None, None, None


def create_minimax_wrapper(
    model_name: str = DEFAULT_MINIMAX_MODEL,
    region: str = DEFAULT_MINIMAX_REGION,
    transport: str = DEFAULT_MINIMAX_TRANSPORT,
    api_key: str | None = None,
    **kwargs: Any,
) -> MiniMaxLlmWrapper:
    resolved_api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
    return MiniMaxLlmWrapper(
        api_key=resolved_api_key,
        model_name=model_name,
        region=region,
        transport=transport,
        **kwargs,
    )
