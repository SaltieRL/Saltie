# Wraps bakkes_repl in a subprocess and exposes it via the rcon() function

from subprocess import Popen, PIPE, TimeoutExpired
import threading
import os
import sys
import atexit

###### Public interface ######

def rcon(command):
    ''' Sends the given command to BakkesMod via bakkes_repl websocket '''
    global repl_process
    message = (command + '\n').encode('utf-8')
    try:
        repl_process.stdin.write(message)
        repl_process.stdin.flush()
    except Exception as e:
        global have_notified_about_repl_dying
        if have_notified_about_repl_dying:
            return
        have_notified_about_repl_dying = True
        raise Exception("=== bakkes_repl.py died ===")

def convert_tick_packet_to_command(game_tick_packet):
    commands = []
    game_info = game_tick_packet.gameInfo
    ball = game_tick_packet.gameball
    commands.append('ball location {} {} {}'.format(ball.Location.X, ball.Location.Y, ball.Location.Z))
    commands.append('ball velocity {} {} {}'.format(ball.Velocity.X, ball.Velocity.Y, ball.Velocity.Z))
    commands.append('ball rotation {} {} {}'.format(ball.Rotation.Pitch, ball.Rotation.Yaw, ball.Rotation.Roll))
    commands.append('ball angularvelocity {} {} {}'.format(ball.AngularVelocity.X, ball.AngularVelocity.Y, ball.AngularVelocity.Z))
    for i, car in enumerate(game_tick_packet.gamecars[:game_tick_packet.numCars]):
        commands.append('player {} location {} {} {}'.format(i, car.Location.X, car.Location.Y, car.Location.Z))
        commands.append('player {} velocity {} {} {}'.format(i, car.Velocity.X, car.Velocity.Y, car.Velocity.Z))
        commands.append('player {} rotation {} {} {}'.format(i, car.Rotation.Pitch, car.Rotation.Yaw, car.Rotation.Roll))
        commands.append('player {} angularvelocity {} {} {}'.format(i, car.AngularVelocity.X, car.AngularVelocity.Y, car.AngularVelocity.Z))
    return ';'.join(commands)

###### End public interface ######

def print_file(f):
    for line in f: print(line.decode('utf-8').rstrip())

def start_gui_subprocess():
    # Create a new process such that asyncio doesn't complain about not being in the main thread
    global read_out
    global read_err
    bakkes_repl_dir = os.path.dirname(os.path.realpath(__file__))

    repl_process = Popen(
        'python bakkes_repl.py --silent',
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        cwd=bakkes_repl_dir,
    )
    atexit.register(lambda: repl_process.kill())  # behave like a daemon
    read_out = threading.Thread(target=print_file, args=[repl_process.stdout], daemon=True)
    read_out.start()
    read_err = threading.Thread(target=print_file, args=[repl_process.stderr], daemon=True)
    read_err.start()
    return repl_process

have_notified_about_repl_dying = False
repl_process = start_gui_subprocess()


