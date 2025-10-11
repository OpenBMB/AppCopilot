import logging
import os
import sys

from datetime import datetime
from pathlib import Path


# utils
def generate_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# Define a constant for the shared logger name
SHARED_LOGGER_NAME = "api_app_logger"


def setup_logging_config():
    """
    Configures and returns a shared logger instance.
    Ensures that the configuration is done only once to avoid duplicate handlers.
    """
    # Attempt to get the existing logger instance
    logger = logging.getLogger(SHARED_LOGGER_NAME)

    # Check if the logger has already been configured with handlers; if so, return it directly.
    # This prevents adding handlers multiple times if setup_logging_config() is called more than once.
    if logger.handlers:
        return logger

    # If the logger is not configured, proceed with configuration.
    logger.setLevel(logging.INFO)  # Set the minimum logging level for the logger.

    log_dir_home = Path.cwd() / "api_task" / "log"
    log_file_path = None

    try:
        log_dir_home.mkdir(parents=True, exist_ok=True)
        potential_log_file_path = log_dir_home / f"api_{generate_timestamp()}.log"

        # Try to create or open the file to check for write permissions.
        with potential_log_file_path.open("a", encoding="utf-8") as f:
            f.write("")  # Try to write an empty string to ensure writability.
        log_file_path = str(potential_log_file_path)
    except OSError as e:
        print(
            f"Warning: Could not create or write to log file at '{log_dir_home}'. Using /tmp instead. Error: {e}",
            file=sys.stderr,
        )
        tmp_dir = Path("/tmp")
        log_file_path = str(tmp_dir / "gpu_monitor.log")
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            with (tmp_dir / "gpu_monitor.log").open("a", encoding="utf-8") as f:
                f.write("")  # Try again to ensure writability in /tmp.
        except OSError as e:
            print(
                f"Critical Warning: Could not create or write to log file in /tmp. File logging will be disabled. Error: {e}",
                file=sys.stderr,
            )
            log_file_path = None  # Could not write to file, disabling file logging.

    # Define the log format.
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s")

    # File Handler
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(
            logging.INFO
        )  # The file handler records INFO level and above.
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(
        sys.stdout
    )  # Explicitly direct output to stdout.
    console_handler.setLevel(
        logging.WARNING
    )  # The console handler only records WARNING level and above.
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Disable propagation to prevent log events from being passed to the root logger, which would cause duplicate output.
    logger.propagate = False

    return logger


if __name__ == "__main__":
    setup_logging_config()
