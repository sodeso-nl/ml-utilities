import subprocess
import os


def start_tensorboard(dir_name='./logs'):
    subprocess.Popen(["tensorboard", "--logdir", dir_name])
    print("http://localhost:6006")


def stop_tensorboard():
    os.system("kill $(ps -e | grep 'tensorboard' | awk '{print $1}')")
