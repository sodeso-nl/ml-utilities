import subprocess as _subprocess
import os as _os


def start_tensorboard(target_dir: str = './logs') -> None:
    """Starts the tensorboard application.

    Args:
        target_dir: the target directory containing the logs
    """
    _subprocess.Popen(["tensorboard", "--logdir", target_dir])
    print("http://localhost:6006")


def stop_tensorboard() -> None:
    """Stops the tensorboard application"""
    _os.system("kill $(ps -e | grep 'tensorboard' | awk '{print $1}')")
