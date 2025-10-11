# Add two API modules: Bilibili-related video operations and automatic email sending assistant
import sys
import smtplib
import os
import json
import subprocess
import mimetypes

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email_validator import validate_email, EmailNotValidError
from typing import Optional, List, Tuple
from api_task.api_logging import setup_logging_config
from pathlib import Path

logger = setup_logging_config()

# pip3 install bilibili-api-python
from bilibili_api import Credential, user, sync, video, search

CONFIG_PATH = "./api_task/config.json"


def get_config_block(config_path, config_name: str) -> dict:
    with open(config_path, "r", encoding="utf8") as file:
        data = json.load(file)
    for block in data:
        if block.get("config_name") == config_name:
            return block
    raise ValueError(f"Config block with config_name '{config_name}' not found.")


# feat: add adb pull command for pulling files into local for email.
def pull_file_from_android(
    device_file_path: str = "/sdcard/DCIM/Camera/",
    local_destination_dir: Path | str | None = None,
) -> Path | None:
    """
    Use the ADB pull command to pull files from an Android device to local.

    Args:
        device_file_path (str): Full path of the file on the Android device (e.g. /sdcard/Documents/my_file.txt).
        local_destination_dir (Path | str | None): Target directory to store the file on the local computer.
                                                   If None, defaults to 'api_task/log/android_file' under the current working directory.

    Returns:
        Path | None: If successful, returns the full Path object of the pulled local file; otherwise returns None.
    """
    # Handle default value and type for local_destination_dir
    if local_destination_dir is None:
        local_destination_dir = Path.cwd() / "api_task" / "log" / "android_file"
    elif isinstance(local_destination_dir, str):
        local_destination_dir = Path(local_destination_dir)

    try:
        os.makedirs(local_destination_dir, exist_ok=True)
        logger.info(f"Created local destination directory: {local_destination_dir}")
    except OSError as e:
        logger.error(
            f"Failed to create local destination directory {local_destination_dir}: {e}"
        )
        return None

    file_name = Path(device_file_path).name
    pulled_local_path = local_destination_dir / file_name  # Compose the full local path

    # Build the ADB pull command
    # Note: If the target path of adb pull is a directory, it will create a file with the same name in that directory
    command = [
        "adb",
        "pull",
        device_file_path,
        str(local_destination_dir),
    ]  # Convert Path object to string for subprocess
    logger.info(f"Attempting to pull file: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            check=True,  # Raises CalledProcessError if the command returns a non-zero exit code
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode output as text
        )

        # Verify if the file was actually pulled to local
        if pulled_local_path.exists():
            logger.info(
                f"Successfully pulled {device_file_path} to {pulled_local_path}"
            )
            logger.debug(f"ADB stdout: {result.stdout.strip()}")
            logger.debug(f"ADB stderr: {result.stderr.strip()}")
            return pulled_local_path
        else:
            logger.error(
                f"Failed to pull file {device_file_path}. Local file not found at {pulled_local_path} after pull."
            )
            logger.debug(f"ADB stdout: {result.stdout.strip()}")
            logger.debug(f"ADB stderr: {result.stderr.strip()}")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Failed to pull file {device_file_path}. ADB Error: {e.stderr.strip()}"
        )
        logger.debug(f"ADB stdout: {e.stdout.strip()}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during file pull: {e}")
        return None


class EmailSender:
    """
    A class to send emails with optional attachments using SMTP.

    Attributes:
        email_config_path (str): Path to the JSON config file containing sender and SMTP info.
        sender_email (str): Sender's email address.
        sender_password (str): Sender's email password.
        smtp_server (str): SMTP server address.
        smtp_port (int): SMTP server port.
    """

    def __init__(self, email_config_path: str = CONFIG_PATH):
        """
        Initialize the EmailSender by loading configuration from a JSON file.

        Args:
            email_config_path (str): Path to the JSON config file.
        """
        self.email_config_path = email_config_path
        self.sender_email, self.sender_password = self._get_usr_config()
        self.smtp_server, self.smtp_port = self._get_smtp_config()
        self.logger = setup_logging_config()

    def _get_usr_config(self) -> Tuple[str, str]:
        block = get_config_block(CONFIG_PATH, "email")
        return str(block["sender_email"]).strip(), str(block["sender_password"]).strip()

    def _get_smtp_config(self) -> Tuple[str, int]:
        block = get_config_block(CONFIG_PATH, "email")
        return str(block["smtp_server"]).strip(), block["smtp_port"]

    def _attach_file(self, msg: MIMEMultipart, file_path: str):
        """
        Attach a file to the email message if the file exists.

        Args:
            msg (MIMEMultipart): The email message object.
            file_path (str): Path to the file to attach.
        """
        if os.path.exists(file_path):
            try:
                # Try to get MIME type based on file extension
                ctype, encoding = mimetypes.guess_type(file_path)
                if ctype is None or encoding is not None:
                    # Fallback to generic type if unable to guess or encoding exists
                    ctype = "application/octet-stream"

                maintype, subtype = ctype.split("/", 1)

                with open(file_path, "rb") as attachment:
                    part2 = MIMEBase(maintype, subtype)
                    part2.set_payload(attachment.read())

                encoders.encode_base64(part2)

                filename = os.path.basename(file_path)
                part2.add_header(
                    "Content-Disposition",
                    f"attachment; filename*=utf-8''{filename}",
                )
                part2.add_header(
                    "Content-Disposition", f"attachment; filename*=utf-8''{filename}"
                )
                # # Note: If the filename contains non-ASCII characters, there may be garbled text or replacements here
                # part2.add_header(
                #     "Content-Disposition",
                #     f"attachment; filename=\"{filename}\"" # Use double quotes to enclose the filename
                # )
                msg.attach(part2)
            except Exception as e:
                self.logger.warning(f"Could not attach file {file_path}: {e}")
        else:
            self.logger.warning(
                f"Warning: Attachment file not found at {file_path}. Skipping this attachment."
            )

    def send_mail(
        self,
        receiver_email: str,
        subject: str = "hello world",
        body: str = "hello world, just for fun!",
        attach_local_file_path: Optional[str | List[str]] = None,
        attach_android_file_path: Optional[str | List[str]] = None,
    ):
        """
        Send an email with optional attachments.

        Args:
            receiver_email (str): Recipient's email address.
            subject (str): Email subject.
            body (str): Email body (plain text).
            attach_file_path (Optional[str | List[str]]): Path(s) to files to attach.
            sender_email (Optional[str]): Override sender email (default: from config).
            sender_password (Optional[str]): Override sender password (default: from config).

        Raises:
            smtplib.SMTPAuthenticationError: If authentication fails.
            Exception: For other errors during sending.
        """
        # Validate sender email
        try:
            validate_email(self.sender_email, check_deliverability=False)
        except EmailNotValidError as e:
            self.logger.error(f"Sender email '{self.sender_email}' is not valid: {e}")
            return

        # --- Create the email ---
        msg = MIMEMultipart()
        msg["From"] = f"<{self.sender_email}>"
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        msg.add_header("Content-Type", 'multipart/mixed; charset="utf-8"')

        # --- Add attachments ---
        # --- Add attachments on Android
        if attach_android_file_path is not None:
            if isinstance(attach_android_file_path, str):
                attach_android_file_path = [attach_android_file_path]
            # all transfer into local devices
            new_path = [
                str(pull_file_from_android(device_file_path=an_file_path))
                for an_file_path in attach_android_file_path
            ]
            for file_path_n in new_path:
                self._attach_file(msg, file_path_n)

        if attach_local_file_path is not None:
            if isinstance(attach_local_file_path, str):
                attach_local_file_path = [attach_local_file_path]
            for file_path in attach_local_file_path:
                self._attach_file(msg, file_path)

        # --- Send the email ---
        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            self.logger.info("Email sent successfully!")
            self.logger.info(
                f"Message sent from {self.sender_email} to {receiver_email}"
            )
        except smtplib.SMTPAuthenticationError:
            self.logger.error("Failed to send email!")
            self.logger.error(
                "Authentication failed: Please check if the sender email address and password/authorization code are correct."
            )
        except Exception as e:
            self.logger.error("Failed to send email!")
            self.logger.error(f"An error occurred: {e}")


class BilibiliOperator:
    def __init__(self) -> None:
        self._load_config()
        self.credential = Credential(
            sessdata=self.SESSDATA,
            bili_jct=self.BILI_JCT,
            buvid3=self.BUVID3,
            dedeuserid=self.user_id,
        )
        self.logger = setup_logging_config()

    def _load_config(self, config_name: str = "bilibili"):
        block = get_config_block(CONFIG_PATH, config_name)
        self.SESSDATA = block["SESSDATA"]
        self.BILI_JCT = block["BILI_JCT"]
        self.BUVID3 = block["BUVID3"]
        self.user_id = block["dedeuserid"]

    async def search_video(self, keyword: str):
        # return await search.search_by_type(
        #     keyword=keyword,
        #     search_type=search.SearchObjectType.VIDEO,
        #     order_type=search.OrderVideo.SCORES,
        #     page=1,
        #     time_range=10
        # )
        return await search.search(keyword)
        # Returns a relatively raw dictionary

    async def get_info(self, video_bvid):
        try:
            v = video.Video(bvid=video_bvid, credential=self.credential)
            info = await v.get_info()
            self.logger.info(f"Video Title: {info.get('title', 'N/A')}")
            self.logger.info(
                f"Current like status of video {video_bvid}: {info.get('like', 'N/A')} (Bilibili API's 'like' field usually reflects your like status)"
            )

        except Exception as e:
            self.logger.error(f"Error while fetching video info {video_bvid}: {e}")

    async def like(self, video_bvid: str, mode: bool = True):
        """
        Likes or unlikes a Bilibili video.

        Args:
            video_bvid (str): The BV ID of the video.
            mode (bool): True for liking (default), False for unliking.
        """
        # Determine the action for print statements
        action_text = "liking" if mode else "unliking"
        like_status_for_api = mode  # v.like(True) for like, v.like(False) for unlike

        self.logger.info(f"Attempting to {action_text} video: {video_bvid}")

        try:
            # Instantiate Video object with the BVID and credentials
            v = video.Video(bvid=video_bvid, credential=self.credential)

            # Await the asynchronous get_info() call
            info = await v.get_info()

            # Some basic information about the video, can be commented out for silent operation
            self.logger.info(f"Video Title: {info.get('title', 'N/A')}")
            self.logger.info(
                f"Current like status of video {video_bvid}: {info.get('like', 'N/A')} (Bilibili API's 'like' field usually reflects your like status)"
            )

            # Await the asynchronous like/unlike call
            await v.like(like_status_for_api)  # Pass True for like, False for unlike

            self.logger.info(
                f"Task for {action_text} video {video_bvid} has completed successfully."
            )

        except Exception as e:
            self.logger.error(f"Error while {action_text} video {video_bvid}: {e}")

    async def coin(self, bvid: str, num_coins: int = 1, select_like: bool = True):
        """
        Coins a Bilibili video.

        Args:
            bvid (str): The BV ID of the video to coin.
            num_coins (int): Number of coins to give (1 or 2). Defaults to 1.
                             You usually have a daily limit of 2 coins.
            select_like (bool): Whether to also like the video while coining. Defaults to True.
        """
        if num_coins not in [1, 2]:
            self.logger.error("Error: You can only give 1 or 2 coins per video.")
            return

        self.logger.info(f"Attempting to coin video: {bvid} with {num_coins} coin(s).")
        if select_like:
            self.logger.info("Simultaneously liking the video.")

        try:
            # Instantiate Video object with the BVID and credentials
            v = video.Video(bvid=bvid, credential=self.credential)

            # Perform the coin action
            # The coin method takes num_coins and select_like (True/False to also like)
            result = await v.pay_coin(num_coins, select_like)

            if result:
                self.logger.info(
                    f"Successfully coined video {bvid} with {num_coins} coin(s)."
                )
                if select_like:
                    self.logger.info(f"Video {bvid} also liked successfully.")
            else:
                self.logger.warning(
                    f"Failed to coin video {bvid}. Result: {result}"
                )  # Check API response if False

        except Exception as e:
            self.logger.error(f"An error occurred while coining video {bvid}: {e}")

    async def triple_interation(self, bvid: str):
        """
        Performs "One-click triple" (Like, Coin, Favorite) on a Bilibili video.

        Args:
            bvid (str): The BV ID of the video to interact with.

        Returns:
            dict: The result of the triple interaction API call, or None if an error occurred.
        """
        self.logger.info(
            f"--- Attempting operation: 'One-click triple' on video {bvid} ---"
        )
        try:
            # Instantiate Video object
            v = video.Video(bvid=bvid, credential=self.credential)
            # Perform triple interaction
            result = await v.triple()
            # Check API return result
            if result and result.get("code") == 0:
                self.logger.info(
                    f"  Task succeeded: 'One-click triple' on video {bvid} completed successfully."
                )
            else:
                self.logger.warning(
                    f"  Operation failed: 'One-click triple' on video {bvid} failed. API returned: {result}"
                )
            return result

        except Exception as e:
            self.logger.error(
                f"  Operation failed: Error occurred during 'One-click triple' on video {bvid}: {e}"
            )
            return None

    async def get_user_info(self, uid: str):
        """
        Retrieves detailed information for a Bilibili user.

        Args:
            uid (str): The User ID (UID) of the Bilibili user.

        Returns:
            dict: A dictionary containing the user's information.
        """
        self.logger.info(f"--- Attempting operation: Get info for user UID: {uid} ---")
        try:
            u = user.User(uid=uid, credential=self.credential)
            info = await u.get_user_info()
            self.logger.info(
                f"  Task succeeded: Successfully retrieved info for user '{info.get('name', 'N/A')}' (UID: {uid})."
            )
            self.logger.info(
                f"  Gender: {info.get('sex', 'N/A')}, Level: LV{info.get('level', 'N/A')}"
            )
            self.logger.info(
                f"  Followers: {info.get('follower', 'N/A')}, Following: {info.get('following', 'N/A')}"
            )
            return info
        except Exception as e:
            self.logger.error(
                f"  Operation failed: Error occurred while getting info for user UID {uid}: {e}"
            )
            return None

    async def follow(self, uid: str, mode: bool = True):
        """
        Follows or unfollows a Bilibili user.

        Args:
            uid (str): The User ID (UID) of the Bilibili user.
            mode (bool): True to follow (default), False to unfollow.
        """
        action_text = "following" if mode else "unfollowing"
        relation_type = (
            user.RelationType.SUBSCRIBE if mode else user.RelationType.UNSUBSCRIBE
        )

        self.logger.info(f"--- Attempting operation: {action_text} user UID: {uid} ---")
        try:
            # Instantiate User object
            u = user.User(uid=uid, credential=self.credential)
            # Perform follow/unfollow operation
            await u.modify_relation(relation_type)
            self.logger.info(
                f"  Task succeeded: Successfully {action_text} user UID: {uid}."
            )
        except Exception as e:
            self.logger.error(
                f"  Operation failed: Error occurred while {action_text} user UID {uid}: {e}"
            )
