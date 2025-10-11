import time
import re
import sys
import os
import numpy as np
import argparse
import speech_recognition as sr
from datetime import datetime

sys.path.append(os.getcwd())
# other modules
# ! attention: use absolute path
from adb_utils import setup_device, change_ui_settings
from log.log_recorder import record_task, TaskLogger
from log.log_replay import replay_log
from log.experience_pool import ExperiencePool
from wrappers.cpm_wrapper import MiniCPMWrapper
from wrappers.constants import AVAILABLE_TASKS
from audio.audio_play import play_random_audio, VoiceType
from audio.tts import run_tts
from user.ocr_service import OCRService
from multi_step.multi_step_execution import is_need_multi_step, extract_info


class TaskRecognizer:
    """Handles speech recognition for user tasks."""

    def __init__(self, max_retries=3, fallback_to_text=True):
        self.recognizer = sr.Recognizer()
        self.max_retries = max_retries
        self.fallback_to_text = fallback_to_text

    def get_task_by_voice(self):
        """Attempts to get a task description via voice input."""
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"[{attempt}/{self.max_retries}] Please speak your task:")
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(source)
                task = self.recognizer.recognize_google(audio, language="zh-CN")
                print(f"Task recognized: {task}")
                return task
            except sr.UnknownValueError:
                print("❌ Could not understand your speech.")
            except sr.RequestError as e:
                print(f"❌ Speech recognition service error: {e}")
                break

        if self.fallback_to_text:
            task = input("Please enter your task manually: ")
            return task
        else:
            raise RuntimeError(
                "Failed to recognize voice input and fallback is disabled."
            )


class GUITaskExecutor:
    """
    Orchestrates the execution of GUI automation tasks,
    handling device interaction, LLM calls, and task logging.
    """

    def __init__(
        self,
        ocr_service: OCRService,
        enable_audio: bool = False,
        enable_vision_parser: bool = False,
        read_final_page: bool = False,
        return_result: bool = False,
        serial: str = None,
    ):
        self.ocr_service = ocr_service
        self.enable_audio = enable_audio
        self.enable_vision_parser = enable_vision_parser
        self.read_final_page = read_final_page
        self.return_result = return_result
        self.serial = serial
        self.device = None
        self.llm = None

    def _play_random_audio(self, type: VoiceType):
        if self.enable_audio:
            play_random_audio(voice_type=type)

    @record_task
    def run_task(
        self, query: str, enable_experience: bool = False, logger: TaskLogger = None
    ):
        """
        Main method to run a GUI automation task.
        Uses @record_task decorator for logging.
        """
        # Set up device and change UI settings
        change_ui_settings(mode="close")
        self.device = setup_device(self.serial, audio_enable=self.enable_audio)

        # Start OCR service
        self.ocr_service.start(self.device)

        # Initialize the LLM wrapper
        self.model_name = "AgentCPM-GUI"
        self.llm = MiniCPMWrapper(
            model_name=self.model_name,
            temperature=1,
            use_history=True,
            history_size=2,
            enable_vision_parser=self.enable_vision_parser,
        )
        is_finish = False
        result = None
        response = None
        action = None

        # check experience
        if enable_experience:
            is_finish = self._load_experience(query)

        # If no matching query or if experience matching is disabled, proceed with the task
        if not is_finish:
            # judge whether it needs multi steps
            self._play_random_audio(VoiceType.COPY)
            is_need_multisteps, step1, step2 = is_need_multi_step(query)

            # requires multi steps
            is_finish, result, response, action = (
                self._multi_execution(query, step1, step2, logger=logger)
                if is_need_multisteps
                else self._single_execution(query, logger=logger)
            )

            self._play_random_audio(VoiceType.FINISH)

            if self.read_final_page:
                self._read_final_page_content()

        # return is_finish, result, response, action
        if self.return_result:
            return is_finish, result, response, action
        else:
            return is_finish, None, None, None

    def _execution(
        self, query: str, last_result_: dict = None, logger: TaskLogger = None
    ):
        """Handles a single step of execution."""
        is_finish = False
        result = None
        _response = None

        while not is_finish:
            text_prompt = query
            screenshot = self.device.screenshot(1120)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "./user/ocr_output"
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"temp_{timestamp}.png")
            screenshot.save(file_path)

            if last_result_ is not None:
                _response = self.llm.predict_mm(
                    text_prompt, [np.array(screenshot)], last_result=last_result_
                )
            else:
                _response = self.llm.predict_mm(text_prompt, [np.array(screenshot)])

            response_text = _response[2]
            action = _response[3]
            if isinstance(action, dict) and "RESULT" in action:
                result = action["RESULT"]
            print(f"action: {action}")

            is_finish = self.device.step(action)
            if logger:
                logger.record_step(
                    screenshot=screenshot, action=action, response=str(response_text)
                )
            time.sleep(2.5)
        if self.return_result:
            return is_finish, result, _response, action
        else:
            return is_finish, None, None, None

    def _multi_execution(self, query, step1, step2, logger: TaskLogger = None):
        """Executes a multi-step task."""
        print("Multi-step execution detected.")
        print(f"Executing first step: {step1}")

        # First step execution
        is_finish_step1, _, _, _ = self._execution(step1, logger)

        # Clear history for the second step
        self.llm.clear_history()

        start_time = time.time()
        last_result_ = extract_info(query, logger.get_log_data())
        # get thr last result and execute the next steps

        end_time = time.time()
        print(f"Time cost for extracting information: {end_time - start_time} seconds")
        print(f"last_result: {last_result_}")

        back_action = {"PRESS": "HOME"}
        self.device.step(back_action)
        # return to main
        if logger:
            logger.record_step(
                screenshot=self.device.screenshot(1120),
                action=back_action,
                response="返回主页",
            )

        print(f"Executing second step: {step2}")
        return self._execution(step2, last_result_, logger=logger)

    def _single_execution(self, query, logger: TaskLogger = None):
        """Executes a single-step task."""
        print("Single-step execution.")
        return self._execution(query, logger=logger)

    def _read_final_page_content(self):
        """Reads and speaks the content of the final page."""
        print("Reading final page content...")

        screenshot = self.device.screenshot(1120)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "./user/ocr_output"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"temp_{timestamp}.png")
        screenshot.save(file_path)
        ocr_results = self.ocr_service.process_ocr(self.device)

        def extract_chinese_text(ocr_data):
            all_text = "".join(item["text"] for item in ocr_data if "text" in item)
            chinese_only = re.findall(r"[\u4e00-\u9fa5]+", all_text)
            return "".join(chinese_only)

        chinese_text = extract_chinese_text(ocr_results)
        if ocr_results:
            print("Final page OCR results:", chinese_text)
        else:
            print("No OCR results found.")
        run_tts(chinese_text, output="assets/audio/voice_temp/final_page.mp3")
        play_random_audio(VoiceType.TEMP)

    def _load_experience(self, query):
        # load experience pool and match query
        self._play_random_audio(VoiceType.COPY)
        print("Experience matching enabled.")
        print("Loading experience pool...")
        is_finish = False

        # Initialize experience pool
        pool = ExperiencePool()
        # Match the query against the experience pool
        mapped_query = pool.match_query(query)

        # If a matching query is found, replay the log
        if mapped_query != "no_match":
            print(f"Query hit in experience pool: {mapped_query}")
            log_path = pool.map[mapped_query]
            print(f"Replaying log from: {log_path}")
            replay_log(log_path)

            # * finish the task
            is_finish = True
            self._play_random_audio(VoiceType.FINISH)
        return is_finish


class ArgumentHandler:
    """Handles parsing and validation of command-line arguments."""

    def __init__(self):
        self.parser = self._setup_parser()

    def _setup_parser(self):
        """
        Configures the argument parser with all supported command-line arguments.

        Arguments:
            --predefined-task (str): Name of a predefined task to run.
                                     Choices are derived from `AVAILABLE_TASKS.keys()`.
                                     Ignored if `--custom-task` is specified.
            --custom-task (store_true): Run a custom task directly without selecting from predefined tasks.
            --enable-experience (store_true): Enable experience-based task matching (works in both modes).
            --enable-voice-input (store_true): Enable voice input (only valid with `--custom-task`).
            --show-tasks (store_true): List all available predefined tasks and exit.
            --enable-audio (store_true): Enable audio feedback (e.g., "swipe", "point").
            --enable-vision-parser (store_true): Enable vision parser for point fixing.
            --read-final-page (store_true): Read out final page content after task completion.

        Returns:
            argparse.ArgumentParser: Configured argument parser.
        """
        parser = argparse.ArgumentParser(description="Run GUI automation tasks")
        parser.add_argument(
            "--predefined-task",
            type=str,
            choices=list(AVAILABLE_TASKS.keys()),
            help="Name of a predefined task to run. Ignored if --custom-task is specified.",
        )
        parser.add_argument(
            "--custom-task",
            action="store_true",
            help="Run a custom task directly without selecting from predefined tasks.",
        )
        parser.add_argument(
            "--enable-experience",
            action="store_true",
            help="Enable experience-based task matching (works in both modes).",
        )
        parser.add_argument(
            "--enable-voice-input",
            action="store_true",
            help="Enable voice input (only valid with --custom-task).",
        )
        parser.add_argument(
            "--show-tasks",
            action="store_true",
            help="List all available predefined tasks and exit.",
        )
        parser.add_argument(
            "--enable-audio",
            action="store_true",
            help='Enable audio feedback (e.g., "swipe", "point").',
        )
        parser.add_argument(
            "--enable-vision-parser",
            action="store_true",
            help="Enable vision parser for point fixing.",
        )
        parser.add_argument(
            "--read-final-page",
            action="store_true",
            help="Read out final page content after task completion.",
        )
        return parser

    def parse_args(self):
        """Parses and returns command-line arguments."""
        return self.parser.parse_args()

    def validate_args(self, args):
        """Validates the parsed arguments."""
        if args.predefined_task and args.custom_task:
            raise ValueError(
                "Conflicting arguments: --predefined-task and --custom-task cannot be used together."
            )
        if not args.predefined_task and not args.custom_task and not args.show_tasks:
            raise ValueError(
                "Please specify a task mode: either --predefined-task for predefined tasks or --custom-task for custom task."
            )
        if not args.custom_task and args.enable_voice_input:
            raise ValueError(
                "Invalid argument: --enable-voice-input is only allowed with --custom-task mode."
            )


def main():
    #  ------ main function start here! ------
    # parse args
    arg_handler = ArgumentHandler()
    args = arg_handler.parse_args()

    # start OCR service
    ocr_service = OCRService()

    try:
        arg_handler.validate_args(args)
        if args.show_tasks:
            print("Available tasks:")
            for task_name, task_desc in AVAILABLE_TASKS.items():
                print(f"  - {task_name}: {task_desc}")
            return
    except ValueError as e:
        print(f"Argument error: {e}")
        arg_handler.parser.print_help()
        return

    # start executing, playing welcome audio
    if args.enable_audio:
        play_random_audio(VoiceType.SELF)
        play_random_audio(VoiceType.WELCOME)

    # start Task Executor
    task_executor = GUITaskExecutor(
        ocr_service=ocr_service,
        enable_audio=args.enable_audio,
        enable_vision_parser=args.enable_vision_parser,
        read_final_page=args.read_final_page,
    )

    if args.enable_voice_input and args.custom_task:
        print("Voice input enabled. Please speak your task after the prompt.")
        task_recognizer = TaskRecognizer()
        task_query = task_recognizer.get_task_by_voice()
        task_executor.run_task(
            task_query,
            enable_experience=args.enable_experience,
        )
    elif not args.enable_voice_input and args.custom_task:
        print("Custom task mode selected. Please describe your task in detail.")
        task_query = input("Enter your task description: ")
        task_executor.run_task(
            task_query,
            enable_experience=args.enable_experience,
        )
    elif args.predefined_task:
        print(f"Running predefined task: {args.predefined_task}")
        task_query = AVAILABLE_TASKS[args.predefined_task]
        task_executor.run_task(
            task_query,
            enable_experience=args.enable_experience,
        )

    # stop OCR service
    ocr_service.stop()

    # update experience pool
    # pool = Experience_Pool()
    # pool.update_query()


if __name__ == "__main__":
    main()
