import abc
import numpy as np
from typing import Optional, Any


class LlmWrapper(abc.ABC):
    """Abstract interface for (text only) LLM."""

    @abc.abstractmethod
    def predict(
        self,
        text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.

        Returns:
          Text output, is_safe, and raw output.
        """


class MultimodalLlmWrapper(abc.ABC):
    """Abstract interface for Multimodal LLM."""

    @abc.abstractmethod
    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.
          images: List of images as numpy ndarray.

        Returns:
          Text output and raw output.
        """
