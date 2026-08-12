import os
from pathlib import Path

import requests


MINIMAX_TTS_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1/t2a_v2",
    "cn_zh": "https://api.minimaxi.com/v1/t2a_v2",
}
MINIMAX_SPEECH_MODELS = (
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
)
MINIMAX_AUDIO_FORMATS = ("mp3", "wav", "flac", "pcm")
DEFAULT_MINIMAX_SPEECH_MODEL = "speech-2.8-hd"
DEFAULT_MINIMAX_TTS_REGION = "global_en"
DEFAULT_MINIMAX_VOICE_ID = "male-qn-qingse"


class MiniMaxSpeechClient:
    """Synchronous client for the MiniMax text-to-audio HTTP API."""

    def __init__(
        self,
        api_key,
        region=DEFAULT_MINIMAX_TTS_REGION,
        model=DEFAULT_MINIMAX_SPEECH_MODEL,
        session=None,
        timeout=60,
    ):
        if not api_key:
            raise ValueError("MINIMAX_API_KEY is required for speech synthesis.")
        if region not in MINIMAX_TTS_ENDPOINTS:
            raise ValueError(f"Unsupported MiniMax TTS region: {region}")
        if model not in MINIMAX_SPEECH_MODELS:
            raise ValueError(f"Unsupported MiniMax speech model: {model}")

        self.api_key = api_key
        self.endpoint = MINIMAX_TTS_ENDPOINTS[region]
        self.model = model
        self.session = requests.Session() if session is None else session
        self.timeout = timeout

    def synthesize(
        self,
        text,
        voice_id=DEFAULT_MINIMAX_VOICE_ID,
        audio_format="mp3",
        speed=1.0,
    ):
        if not text or not text.strip():
            raise ValueError("Speech text must not be empty.")
        if len(text) >= 10000:
            raise ValueError("Speech text must be shorter than 10,000 characters.")
        if audio_format not in MINIMAX_AUDIO_FORMATS:
            raise ValueError(f"Unsupported MiniMax audio format: {audio_format}")

        payload = {
            "model": self.model,
            "text": text,
            "stream": False,
            "language_boost": "auto",
            "output_format": "hex",
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed,
                "vol": 1,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": audio_format,
                "channel": 1,
            },
            "subtitle_enable": False,
        }
        response = self.session.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        response_data = response.json()

        base_response = response_data.get("base_resp") or {}
        if base_response.get("status_code") != 0:
            message = base_response.get("status_msg") or "unknown API error"
            raise RuntimeError(f"MiniMax speech synthesis failed: {message}")

        audio_data = response_data.get("data") or {}
        if audio_data.get("status") != 2:
            raise RuntimeError("MiniMax speech synthesis did not complete.")
        encoded_audio = audio_data.get("audio")
        if not isinstance(encoded_audio, str) or not encoded_audio:
            raise RuntimeError("MiniMax speech response did not contain audio data.")

        try:
            return bytes.fromhex(encoded_audio)
        except ValueError as error:
            raise RuntimeError("MiniMax speech response contained invalid audio data.") from error


def run_tts(
    text,
    output="assets/audio/voice_temp/output.mp3",
    *,
    api_key=None,
    region=None,
    model=None,
    voice_id=None,
    audio_format=None,
    session=None,
):
    """Synthesize speech with MiniMax and save it to a local audio file."""
    client = MiniMaxSpeechClient(
        api_key=api_key or os.getenv("MINIMAX_API_KEY", ""),
        region=region
        or os.getenv("MINIMAX_TTS_REGION", DEFAULT_MINIMAX_TTS_REGION),
        model=model
        or os.getenv("MINIMAX_TTS_MODEL", DEFAULT_MINIMAX_SPEECH_MODEL),
        session=session,
    )
    audio = client.synthesize(
        text=text,
        voice_id=voice_id
        or os.getenv("MINIMAX_TTS_VOICE_ID", DEFAULT_MINIMAX_VOICE_ID),
        audio_format=audio_format or Path(output).suffix.lstrip(".").lower() or "mp3",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio)
    print(f"Speech has been saved as {output}")
    return output
