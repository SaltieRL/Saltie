'''
A utility function such that you can test/train in-air controls without ever landing.
'''

import time
from bot_code.trainer.utils.floating_setup import bakkes
from random import random

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

def make_ball_float(location=(200, 0, 500)):
    if not should_call_bakkes('BALL'):
        return
    bakkes.rcon(';'.join([
        'ball location {} {} {}'.format(*location),
        'ball velocity 0 0 0',
    ]))


last_rotation_modification = {}  # player_index -> time of last change of rotation/angular vel
def set_random_pitch_and_pitch_vel_periodically(player_index, period=2.0):
    now = time.clock()
    if now - last_rotation_modification.get(player_index, 0) > period:
        last_rotation_modification[player_index] = now
        set_random_pitch_and_pitch_vel(player_index)

def set_random_pitch_and_pitch_vel(player_index):
    bakkes.rcon(';'.join([
        'player {} rotation {} 0 0'.format(       player_index, (100000 * (random() - 0.5))),
        'player {} angularvelocity 0 {} 0'.format(player_index, (   100 * (random() - 0.5))),
    ]))
