import pytest

from audio.tts import (
    MINIMAX_SPEECH_MODELS,
    MINIMAX_TTS_ENDPOINTS,
    MiniMaxSpeechClient,
    run_tts,
)


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, body):
        self.body = body
        self.request = None

    def post(self, url, **kwargs):
        self.request = {"url": url, **kwargs}
        return FakeResponse(self.body)


def successful_response(audio=b"speech"):
    return {
        "data": {"audio": audio.hex(), "status": 2},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }


def test_registry_contains_current_models_and_regional_endpoints():
    assert MINIMAX_TTS_ENDPOINTS == {
        "global_en": "https://api.minimax.io/v1/t2a_v2",
        "cn_zh": "https://api.minimaxi.com/v1/t2a_v2",
    }
    assert MINIMAX_SPEECH_MODELS == (
        "speech-2.8-hd",
        "speech-2.8-turbo",
        "speech-2.6-hd",
        "speech-2.6-turbo",
        "speech-02-hd",
        "speech-02-turbo",
        "speech-01-hd",
        "speech-01-turbo",
    )


def test_run_tts_uses_china_endpoint_and_writes_decoded_audio(tmp_path):
    session = FakeSession(successful_response())
    output = tmp_path / "result.mp3"

    result = run_tts(
        "Read the final page",
        output=str(output),
        api_key="test-token",
        region="cn_zh",
        model="speech-2.8-turbo",
        voice_id="male-qn-qingse",
        session=session,
    )

    assert result == str(output)
    assert output.read_bytes() == b"speech"
    assert session.request["url"] == "https://api.minimaxi.com/v1/t2a_v2"
    assert session.request["headers"]["Authorization"] == "Bearer test-token"
    assert session.request["timeout"] == 60
    assert session.request["json"] == {
        "model": "speech-2.8-turbo",
        "text": "Read the final page",
        "stream": False,
        "language_boost": "auto",
        "output_format": "hex",
        "voice_setting": {
            "voice_id": "male-qn-qingse",
            "speed": 1.0,
            "vol": 1,
            "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1,
        },
        "subtitle_enable": False,
    }


def test_client_rejects_unsuccessful_api_response():
    session = FakeSession(
        {
            "data": None,
            "base_resp": {"status_code": 1004, "status_msg": "request failed"},
        }
    )
    client = MiniMaxSpeechClient(api_key="test-token", session=session)

    with pytest.raises(RuntimeError, match="request failed"):
        client.synthesize("Read the final page")


def test_client_rejects_missing_or_invalid_audio_data():
    client = MiniMaxSpeechClient(
        api_key="test-token",
        session=FakeSession(
            {
                "data": {"audio": "not-hex", "status": 2},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        ),
    )

    with pytest.raises(RuntimeError, match="invalid audio data"):
        client.synthesize("Read the final page")
