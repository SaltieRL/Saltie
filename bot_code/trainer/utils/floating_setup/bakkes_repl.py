"""BakkesMod REPL.
A small script that forwards input to the injected bakkes_mod
which hosts a websocket server.

Usage:
  bakkes_repl.py
  bakkes_repl.py (-s | --silent)

Options:
  -h --help     Show this screen.
  -s --silent   Do not prompt for input.
"""
from docopt import docopt
import asyncio
import websockets


# bakkes websocket documentation: https://docs.google.com/document/d/1HVMrN1hkrt7BiSgFoQO9C-vNUt9RfWzbP1DlqDzMUWI/edit#
bakkes_server = 'ws://127.0.0.1:9002'
rcon_password = 'password'  # http://devhumor.com/content/uploads/images/November2016/optimism.jpg


async def main_loop():
    global get_input
    async with websockets.connect(bakkes_server) as websocket:
        await websocket.send('rcon_password ' + rcon_password)
        auth_status = await websocket.recv()
        assert auth_status == 'authyes'

        while True:
            try:
                line = get_input()
            except EOFError as e:
                return
            except KeyboardInterrupt as e:
                print()
                continue

            await websocket.send(line)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='BakkesMod REPL 1.0')
    if arguments['--silent']:
        get_input = input
    else:
        from prompt_toolkit import prompt
        from prompt_toolkit.history import FileHistory
        history = FileHistory('.bakkes_repl_history.txt')
        get_input = lambda: prompt('bakkes> ', history=history)
    asyncio.get_event_loop().run_until_complete(main_loop())
