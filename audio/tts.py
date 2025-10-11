import edge_tts
import asyncio


async def text_to_speech(
    text, voice="zh-CN-XiaoxiaoNeural", rate="+0%", output="output.mp3"
):
    """
    Convert Chinese text to a speech MP3 file.

    Args:
        text: The Chinese text to synthesize.
        voice: Voice type (default is "Xiaoxiao", female).
        rate: Speech rate (e.g. '+20%' means 20% faster; '-20%' means slower).
        output: Output MP3 file name.
    """
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(output)
    print(f"Speech has been saved as {output}")


def run_tts(text, output="assets/audio/voice_temp/output.mp3"):
    """
    Run text-to-speech conversion.
    """
    voice = "zh-CN-XiaoxiaoNeural"  # Female voice, recommended
    rate = "+20%"  # 20% faster than standard speed

    asyncio.run(text_to_speech(text, voice=voice, rate=rate, output=output))
    return output
