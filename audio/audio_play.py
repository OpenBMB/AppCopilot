import pygame
import time
import random
import os
from enum import Enum

AUDIO_PATH = "assets/audio"


class VoiceType(Enum):
    COPY = "voice_copy"
    FINISH = "voice_finish"
    POINT = "voice_point"
    PRESS = "voice_press"
    SWIPE = "voice_swipe"
    TYPE = "voice_type"
    WELCOME = "voice_welcome"
    OPENAPP = "voice_openapp"
    ENTERSEARCH = "voice_entersearch"
    ENTERPAGE = "voice_enterpage"
    SELF = "voice_self"
    TEMP = "voice_temp"
    INPUTTEXT = "voice_inputtext"


def play_random_audio(voice_type: VoiceType):
    """
    Plays a random MP3 audio file from the specified voice type folder.

    Args:
        voice_type (VoiceType): The type of voice/audio to play. Must be a member of the VoiceType enum.

    Behavior:
        - Selects a random MP3 file from the corresponding folder under AUDIO_PATH.
        - Plays the selected audio file using pygame.
        - Waits until playback is finished before quitting the mixer.
        - Prints an error message if the folder or MP3 files are not found, or if the voice type is invalid.
    """
    if voice_type not in VoiceType:
        print(f"无效的音频类型: {voice_type}")
        return
    voice_folder = voice_type.value
    folder_path = os.path.join(AUDIO_PATH, voice_folder)
    if not os.path.exists(folder_path):
        print(f"音频文件夹不存在: {folder_path}")
        return
    mp3_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp3")]
    if not mp3_files:
        print("未找到任何 MP3 文件！")
        return

    selected_file = random.choice(mp3_files)
    full_path = os.path.join(folder_path, selected_file)

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(full_path)
        pygame.mixer.music.play()

        # wait for the audio to be finished
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    finally:
        # ! important
        pygame.mixer.quit()
