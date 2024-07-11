"""
Use this script to check if the GPT-4 inference script is running.
If not running and not finished, kick it off.
"""

import os
import subprocess
import time

# Set run params
# Be sure to set the right CHECK_DIR
SCRIPT_NAME = "gpt4_in_context.py"
SCRIPT_PATH = "./" + SCRIPT_NAME
CHECK_DIR = "DICE_output_16_in_context_kate_True_ssbbu_pooler_disease"
WAIT_TIME = 300


def is_script_running(script_name):
    process_list = subprocess.Popen(
        ["pgrep", "-fl", script_name], stdout=subprocess.PIPE
    )
    output, _ = process_list.communicate()
    return script_name in output.decode()


def is_final_file_exist(directory):
    for file in os.listdir(directory):
        if "final" in file:
            return True
    return False


def run_script(script_path):
    subprocess.Popen(["python", script_path])


def main():
    while not is_final_file_exist(CHECK_DIR):
        if not is_script_running(SCRIPT_NAME):
            run_script(SCRIPT_PATH)
        time.sleep(WAIT_TIME)


if __name__ == "__main__":
    main()
