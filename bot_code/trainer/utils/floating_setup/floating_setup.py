'''
A utility function such that you can test/train in-air controls without ever landing.
'''

from collections import defaultdict
import time
from bot_code.trainer.utils.floating_setup import bakkes


# Calling bakkesmod too often may cause a crash.
# Therefore ratelimit it by dropping some calls.
MIN_DELAY_BETWEEN_BAKKES_CALLS = 2 * 1/60.
last_bakkes_call = {}  # player_index -> time of last call
def should_call_bakkes(player_index):
    # Note: this function mutates external state: last_bakkes_call.
    now = time.clock()
    if now - last_bakkes_call.get(player_index, 0) > MIN_DELAY_BETWEEN_BAKKES_CALLS:
        last_bakkes_call[player_index] = now
        return True
    return False

def make_player_float(player_index):
    # Call this every frame to reset the players position
    if not should_call_bakkes(player_index):
        return
    height = 250 + 200*player_index
    # Hopefully this dependence onc bakkesmod will be removed with the new RLBot api
    bakkes.rcon(';'.join([
        'player {} location -300 0 {}'.format(player_index, height),
        'player {} velocity -0 0 10'.format(player_index),
    ]))
