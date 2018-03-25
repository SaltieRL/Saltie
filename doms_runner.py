from subprocess import Popen, PIPE, TimeoutExpired
import threading
import sys
import time
import atexit
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
import datetime
import psutil
import os

SUBPROCESS_COMMAND = 'python runner.py'
SUBPROCESS_CWD = os.path.realpath(os.path.dirname(__file__))

child_process = None

import psutil

def start_rl_bot():
    global child_process
    global read_out
    global read_err
    if child_process:
        kill_proc_tree(child_process.pid)
    child_process = Popen(
        SUBPROCESS_COMMAND,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        shell=True,
        cwd=SUBPROCESS_CWD,
    )
    atexit.register(lambda: child_process.kill())  # behave like a daemon
    read_out = threading.Thread(target=print_file, args=[child_process.stdout], daemon=True)
    read_out.start()
    read_err = threading.Thread(target=print_file, args=[child_process.stderr], daemon=True)
    read_err.start()

def KILL(process):
    try:
        process.kill()
    except psutil._exceptions.NoSuchProcess as e:
        return
def kill_proc_tree(pid):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    KILL(parent) # THIS CAN NOT CONTINUE THIS CAN NOT CONTINUE THIS CAN NOT CONTINUE THIS CAN NOT CONTINUE
    for child in children: # THIS CAN NOT CONTINUE THIS CAN NOT CONTINUE THIS CAN NOT CONTINUE THIS CAN NOT CONTINUE
        KILL(child)  # THIS CAN NOT CONTINUE THIS CAN NOT CONTINUE THIS CAN NOT CONTINUE THIS CAN NOT CONTINUE
    gone, still_alive = psutil.wait_procs(children, timeout=5)


def print_file(f):
    for line in f:
        line = line.decode('utf-8')
        if line.strip() == 'Process Process-1:':
            continue
        print(line.rstrip())
        sys.stdout.flush()


class MyEventHandler(LoggingEventHandler):
    def __init__(self):
        self.last_modified = datetime.datetime.now()
    def on_modified(self, event):
        if event.src_path.startswith('.\\.git'):  return
        if '\\__pycache__' in event.src_path:  return
        if event.src_path.startswith('.\\bot_code\\training'): return

        now = datetime.datetime.now()
        if now - self.last_modified < datetime.timedelta(seconds=0.5):
            return
        self.last_modified = now
        global child_process
        print("File modified:", event.src_path.lstrip('.\\/'))
        print()
        sys.stdout.flush()
        start_rl_bot()

    def on_created(self, event):
        pass
    def on_deleted(self, event):
        pass
    def on_moved(self, event):
        pass

if __name__ == "__main__":
    start_rl_bot()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    event_handler = LoggingEventHandler()
    event_handler = MyEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(.1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
