import subprocess
import datetime
import urllib.parse
import logging
import os
import io
import PIL.Image as Image
from typing import List, Dict, Any, Optional
from audio.audio_play import play_random_audio, VoiceType
from image_hash.hash_find import start_hash_find
from audio.tts import run_tts


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Low‑level helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str], timeout: int = 30) -> bytes:
    """Run a shell command and return raw stdout (raises on non‑zero exit)."""
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)


def _adb_prefix(serial: str | None) -> List[str]:
    return ["adb", "-s", serial] if serial else ["adb"]


def _resize_pillow(origin_img, max_line_res: int = 1120):
    """Resize PIL image so that longest edge ≤ `max_line_res` using Lanczos."""
    w, h = origin_img.size
    if max_line_res is not None:
        max_line = max_line_res
        if h > max_line:
            w = int(w * max_line / h)
            h = max_line
        if w > max_line:
            h = int(h * max_line / w)
            w = max_line
    return origin_img.resize((w, h), resample=Image.Resampling.LANCZOS)

def _encode_text_for_adb(text: str) -> str:
    """Encode text for adb shell input.  URL‑encode spaces as %s."""
    def _esc(ch: str) -> str:
        if ord(ch) < 128 and ch != " ":
            return ch
        if ch == " ":
            return "%s"
        return f"\\u{ord(ch):04x}"
    return "".join(_esc(c) for c in text)

def _encode_ascii_for_adb(text: str) -> str:
    """Encode ASCII‑only string for `adb shell input text …` (spaces→%s)."""
    return text.replace(" ", "%s")


# ---------------------------------------------------------------------------
# AndroidDevice class
# ---------------------------------------------------------------------------

class AndroidDevice:
    """Encapsulates a single, already‑connected Android handset."""

    _yadb_pushed: bool = False
    _yadb_local: str = os.path.join(os.path.dirname(__file__), "yadb/yadb")

    def __init__(self, serial: str | None, audio_enable=False):
        self.serial: str | None = serial
        self.width: int = 0
        self.height: int = 0
        self.last_req_time: datetime.datetime = datetime.datetime.now()
        self.audio_enable = audio_enable

    # ---------- internal ----------
    def _adb(self, *args: str, timeout: int = 30) -> bytes:
        return _run(_adb_prefix(self.serial) + list(args), timeout)

    def _ensure_yadb(self):
        if AndroidDevice._yadb_pushed:
            return
        if not os.path.exists(AndroidDevice._yadb_local):
            raise FileNotFoundError(f"yadb helper not found: {AndroidDevice._yadb_local}")
        self._adb("push", AndroidDevice._yadb_local, "/data/local/tmp")
        AndroidDevice._yadb_pushed = True
        logger.info("yadb pushed to device for Unicode input support")

    # ---------- public API ----------
    def refresh_resolution(self) -> None:
        """Query and cache `wm size` (sets .width / .height)."""
        raw = self._adb("shell", "wm", "size").decode()
        try:
            size_line = raw.split("Physical size: ")[1].splitlines()[0]
            self.width, self.height = map(int, size_line.split("x"))
            logger.info("Device %s resolution: %dx%d", self.serial or "<default>",
                        self.width, self.height)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse wm size output: {raw}") from exc

    # -------------------------------------------------------------------
    # Step: execute user action
    # -------------------------------------------------------------------
    def step(self, data: Dict[str, Any]) -> None:
        """Execute a control step on the device (tap/swipe/key/text/clear)."""
        logger.debug("Step: %s", data)
        if "POINT" in data:
            self._handle_point(data)
        if "PRESS" in data:
            self._handle_press(data["PRESS"])
        if "TYPE" in data:
            self._handle_type(data["TYPE"])
        if "CLEAR" in data:
            self._adb("shell", "input", "keyevent", "KEYCODE_CLEAR")
        self.last_req_time = datetime.datetime.now()

        if ("STATUS", "finish") in data.items() or ("STATUS", "impossible") in data.items():
            logger.info("Task finished")
            return True
        return False

    # -------------------------------------------------------------------
    # State snapshot
    # -------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "last_req_time": self.last_req_time.isoformat(),
            "screenshot": self.screenshot(),
        }

    # --- Device state ---------------------------------------------------
    def screenshot(self, max_side: Optional[int] = None) -> Image.Image:
        """Grab screen; return Pillow Image. Optionally down‑scale with user rule."""
        png_bytes = self._adb("exec-out", "screencap", "-p")
        img = Image.open(io.BytesIO(png_bytes))
        if max_side is not None:
            img = _resize_pillow(img, max_side)
        return img

    # =================== private helpers ===================
    def _handle_point(self, data: Dict[str, Any]) -> None:
        search_thershold = 150
        x1, y1 = data["POINT"]
        print("point:",x1,y1)
        x = int(x1 / 1000 * self.width)
        y = int(y1 / 1000 * self.height)
        
        
        if "to" in data:  
            if self.audio_enable:
                play_random_audio(VoiceType.SWIPE)
            if isinstance(data["to"], list):
                x2, y2 = data["to"]
                x2 = int(x2 / 1000 * self.width)
                y2 = int(y2 / 1000 * self.height)
            else:  # directional swipe (up/down/left/right)
                dirs = {
                    "up": (0, -0.15),
                    "down": (0, 0.15),
                    "left": (-0.15, 0),
                    "right": (0.15, 0),
                }
                if data["to"] not in dirs:
                    raise ValueError(f"Invalid swipe direction: {data['to']}")
                dx_ratio, dy_ratio = dirs[data["to"]]
                x2 = int(max(min(x + dx_ratio * self.width, self.width), 0))
                y2 = int(max(min(y + dy_ratio * self.height, self.height), 0))
            dur = str(data.get("duration", 150))
            # Perform swipe
            self._adb("shell", "input", "swipe", str(x), str(y), str(x2), str(y2), dur)
        else:  # simple tap
            if self.audio_enable:
                result = start_hash_find()
                if result == "openapp":
                    play_random_audio(VoiceType.OPENAPP)
                if result == "entersearch":
                    if y1< search_thershold:
                        play_random_audio(VoiceType.ENTERSEARCH)
                    else:
                        play_random_audio(VoiceType.ENTERPAGE)
                if result == "enterpage":
                    if y1< search_thershold:
                        play_random_audio(VoiceType.ENTERSEARCH)
                    else:
                        play_random_audio(VoiceType.ENTERPAGE)
            # Show tap location
            #print("tap:", x,y)
            self._adb("shell", "input", "tap", str(x), str(y))

    def _handle_press(self, key: str) -> None:
        if self.audio_enable:
            play_random_audio(VoiceType.PRESS)
        KEYS = {
            "HOME": "KEYCODE_HOME",
            "BACK": "KEYCODE_BACK",
            "MENU": "KEYCODE_MENU",
            "ENTER": "KEYCODE_ENTER",
            "APPSELECT": "KEYCODE_APP_SWITCH",
            "power": "KEYCODE_POWER",
            "volume_up": "KEYCODE_VOLUME_UP",
            "volume_down": "KEYCODE_VOLUME_DOWN",
            "volume_mute": "KEYCODE_VOLUME_MUTE",
        }
        if key not in KEYS:
            raise ValueError(f"Unknown PRESS value: {key}")
        self._adb("shell", "input", "keyevent", KEYS[key])

    # def _handle_type(self, raw):
    #     decoded = urllib.parse.unquote(raw)
    #     self._adb("shell", "am", "broadcast", '-a', 'ADB_INPUT_TEXT', '--es msg' , decoded)
    #     # self._adb("shell", "input", "text", decoded)

    def _handle_type(self, raw):
        text = urllib.parse.unquote(raw)
        if self.audio_enable:
            run_tts(text, output="assets/audio/voice_inputtext/text.mp3")
            play_random_audio(VoiceType.TYPE)
            play_random_audio(VoiceType.INPUTTEXT)
        if all(ord(c) < 128 for c in text):  # quick ASCII path
            self._adb("shell", "input", "text", _encode_ascii_for_adb(text))
            return
        # Unicode → yadb
        self._ensure_yadb()
        safe = text.replace("'", "'\\''")  # escape sigingle quotes for sh
        cmd = (
            "app_process -Djava.class.path=/data/local/tmp/yadb /data/local/tmp "
            "com.ysbing.yadb.Main -keyboard '%s'" % safe
        )
        self._adb("shell", cmd)

# ---------------------------------------------------------------------------
# Public utility function
# ---------------------------------------------------------------------------

def list_connected_devices() -> List[str]:
    """列出所有已连接的ADB设备"""
    lines = _run(["adb", "devices"]).decode().strip().splitlines()[1:]
    return [l.split()[0] for l in lines if l.strip() and "device" in l]

# 支持指定设备序列号
def setup_device(serial: str | None = None, audio_enable=False) -> AndroidDevice:
    """创建AndroidDevice实例，可指定设备序列号"""
    if serial is None:
        devices = list_connected_devices()
        if not devices:
            raise RuntimeError("No authorised Android device found. Plug in & check adb.")
        if len(devices) > 1:
            logger.warning("Multiple devices detected; defaulting to the first (%s).", devices[0])
        serial = devices[0]
    dev = AndroidDevice(serial, audio_enable=audio_enable)
    dev.refresh_resolution()
    return dev

def change_ui_settings(mode: str = "open"):
    mode = mode.strip()
    if mode == "open":
        subprocess.run("adb shell settings put system pointer_location 1", shell=True)
    elif mode == "close":
        subprocess.run("adb shell settings put system pointer_location 0", shell=True)
    else:
        raise ValueError("Invalid Openmode")

# ---------------------------------------------------------------------------
# Demo – run this file directly to test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = setup_device()
    logger.info("Device ready: serial=%s (%dx%d)", device.serial, device.width, device.height)

    # Example: tap centre, take screenshot
    x = 900
    y = 800
    device.step({"POINT": [x, y]})
    png = device.screenshot()
    target = os.path.join("screenshots", "screencap.png")
    # logger.info("Screenshot saved → %s (%d bytes)", target, len(png))