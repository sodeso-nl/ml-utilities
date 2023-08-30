import subprocess
import os


def start_tensorboard(target_dir: str = './logs') -> None:
    """Starts the tensorboard application.

    Args:
        target_dir: the target directory containing the logs
    """
    subprocess.Popen(["tensorboard", "--logdir", target_dir])
    print("http://localhost:6006")


def stop_tensorboard() -> None:
    """Stops the tensorboard application"""
    os.system("kill $(ps -e | grep 'tensorboard' | awk '{print $1}')")
