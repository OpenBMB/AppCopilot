import json
import os
import logging
import builtins

from datetime import datetime
from functools import wraps
from PIL import Image

class TaskLogger:
    """A context manager for logging task execution.

    This class records the start and end times of a task, its status, and a
    detailed list of steps, including screenshots, actions, and responses.
    Logs are saved in a structured directory format with a JSON file and
    corresponding screenshots.
    """
    
    def __init__(self, log_dir="./log/task_logs_new"):
        """Initializes the TaskLogger.

        Args:
            log_dir (str): The base directory to save logs.
        """
        self.log_dir = log_dir
        self.log_data = {
            "metadata": {
                "start_time": None,
                "end_time": None,
                "status": "running",
                "experience_flag": False, # Changed to boolean for clarity
                "query": ""
            },
            "steps": []
        }
        self.current_step = 0
        self.log_path = None
    
    def __enter__(self):
        """Starts the logging process and sets up the log directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(self.log_dir, timestamp)
        os.makedirs(os.path.join(self.log_dir, "screenshots"), exist_ok=True)
        self.log_data["metadata"]["start_time"] = datetime.now().isoformat()
        logging.info(f"Logger started. Logs will be saved to: {self.log_dir}")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Finalizes the log, saves it to a JSON file, and handles exceptions."""
        success = exc_type is None
        self.log_data["metadata"]["end_time"] = datetime.now().isoformat()

        # Determine if the task was successful based on the last step's status
        last_step_ok = False
        if self.log_data["steps"]:
            last_step_ok = (self.log_data["steps"][-1].get("action", {}).get("STATUS") == "finish")
        
        self.log_data["metadata"]["status"] = "completed" if (last_step_ok and success) else "failed"

        # The 'experience_flag' is initially set to False and can be changed later
        # self.log_data["metadata"]["experience_flag"] = False # This line is redundant as it's set in __init__
        
        self.log_path = os.path.join(self.log_dir, "action_log.json")
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)
        
        if success:
            logging.info(f"Log saved successfully to {self.log_path}")
        else:
            logging.error(f"Task failed. Log saved to {self.log_path}")
        return True  # Suppress the exception from propagating
    
    def record_step(self, screenshot: Image.Image, action: dict, response: dict):
        """Records a single step of the task.

        Args:
            screenshot (Image.Image): The screenshot of the current state.
            action (dict): The action performed in this step.
            response (dict): The response received after the action.
        """
        self.current_step += 1
        step_id = self.current_step
        
        # Save the screenshot
        screenshot_path = os.path.join(
            self.log_dir, "screenshots", f"step_{step_id:03d}.png"
        )
        screenshot.save(screenshot_path)

        # Record the step data
        step_data = {
            "step": step_id,
            "timestamp": datetime.now().isoformat(),
            "screenshot": screenshot_path,
            "action": action,
            "response": response
        }
        self.log_data["steps"].append(step_data)
        logging.info(f"Recorded step {step_id}: Action={action}")

    def get_log_data(self):
        """Returns the current log data dictionary."""
        return self.log_data

def record_task(func):
    """A decorator that automatically logs the execution of a task.

    This decorator initializes a TaskLogger, passes it to the decorated
    function, and handles logging the task's start and end, including
    any exceptions that occur. It also temporarily redirects the built-in
    `print` function to log its output.
    """
    @wraps(func)
    def wrapper(self, query, *args, **kwargs):
        with TaskLogger() as logger:
            # Record initial task information
            logger.log_data["metadata"]["query"] = query
            
            # Save the original print function
            original_print = builtins.print
            
            def print_wrapper(*print_args, **print_kwargs):
                """A wrapper for the print function to also log its output."""
                message = " ".join(str(arg) for arg in print_args)
                original_print(f"{message}")
                logging.info(f"Original print: {message}")
            
            try:
                # Replace the built-in print function with the wrapper
                builtins.print = print_wrapper
                
                # Execute the decorated function, passing the logger instance
                result = func(self, query, *args, logger=logger, **kwargs)
                
                # The result is not logged to avoid serialization issues
                return result
            except Exception as e:
                logging.error(f"Task execution failed: {str(e)}")
                print(f"Error: {str(e)}")
            finally:
                # Restore the original print function
                builtins.print = original_print
    
    return wrapper