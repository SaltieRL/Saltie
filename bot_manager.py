import ctypes
import gzip
from datetime import datetime, timedelta
import importlib
import mmap
import os
import shutil
import time

import numpy as np

import bot_input_struct as bi
import game_data_struct as gd
import rate_limiter
import sys
import traceback

from bot_code.conversions import binary_converter as compressor
from bot_code.conversions.input import input_formatter

OUTPUT_SHARED_MEMORY_TAG = 'Local\\RLBotOutput'
INPUT_SHARED_MEMORY_TAG = 'Local\\RLBotInput'
GAME_TICK_PACKET_REFRESHES_PER_SECOND = 120  # 2*60. https://en.wikipedia.org/wiki/Nyquist_rate
MAX_AGENT_CALL_PERIOD = timedelta(seconds=1.0/30)  # Minimum call rate when paused.
REFRESH_IN_PROGRESS = 1
REFRESH_NOT_IN_PROGRESS = 0
MAX_CARS = 10


class BotManager:

    game_file = None
    model_hash = None
    is_eval = False

    def __init__(self, terminateEvent, callbackEvent, bot_parameters, name, team, index, modulename, gamename, savedata, server_manager):
        self.terminateEvent = terminateEvent
        self.callbackEvent = callbackEvent
        self.bot_parameters = bot_parameters
        self.name = name
        self.team = team
        self.index = index
        self.save_data = savedata
        self.module_name = modulename
        self.game_name = gamename
        self.input_converter = input_formatter.InputFormatter(team, index)
        self.frames = 0
        self.file_number = 1
        self.server_manager = server_manager
        self.input_array = np.array([])
        self.output_array = np.array([])
        self.batch_size = 1000
        self.upload_size = 20

    def load_agent(self, agent_module):
        try:
            agent = agent_module.Agent(self.name, self.team, self.index, bot_parameters=self.bot_parameters)
        except TypeError as e:
            agent = agent_module.Agent(self.name, self.team, self.index)
        return agent

    def run(self):
        # Set up shared memory map (offset makes it so bot only writes to its own input!) and map to buffer
        filename = ""
        buff = mmap.mmap(-1, ctypes.sizeof(bi.GameInputPacket), INPUT_SHARED_MEMORY_TAG)
        bot_input = bi.GameInputPacket.from_buffer(buff)
        player_input = bot_input.sPlayerInput[self.index]
        player_input_lock = (ctypes.c_long).from_address(ctypes.addressof(player_input))

        # Set up shared memory for game data
        game_data_shared_memory = mmap.mmap(-1, ctypes.sizeof(gd.GameTickPacketWithLock), OUTPUT_SHARED_MEMORY_TAG)
        bot_output = gd.GameTickPacketWithLock.from_buffer(game_data_shared_memory)
        lock = ctypes.c_long(0)
        game_tick_packet = gd.GameTickPacket()  # We want to do a deep copy for game inputs so people don't mess with em


        # Create Ratelimiter
        r = rate_limiter.RateLimiter(GAME_TICK_PACKET_REFRESHES_PER_SECOND)
        last_tick_game_time = None  # What the tick time of the last observed tick was
        last_call_real_time = datetime.now()  # When we last called the Agent

        # Find car with same name and assign index
        for i in range(MAX_CARS):
            if str(bot_output.gamecars[i].wName) == self.name:
                self.index = i
                continue

        # Get bot module
        agent_module = importlib.import_module(self.module_name)
        # Create bot from module
        agent = self.load_agent(agent_module)

        if hasattr(agent, 'create_model_hash'):
            self.model_hash = agent.create_model_hash()
        else:
            self.model_hash = 0

        self.server_manager.set_model_hash(self.model_hash)
        last_module_modification_time = os.stat(agent_module.__file__).st_mtime

        if hasattr(agent, 'is_evaluating'):
            self.is_eval = agent.is_evaluating
            self.server_manager.set_is_eval(self.is_eval)

        if self.save_data:
            filename = self.create_file_name()
            print('creating file ' + filename)
            self.create_new_file(filename)
        old_time = 0
        counter = 0

        last_module_modification_time = os.stat(agent_module.__file__).st_mtime

        # Run until main process tells to stop
        while not self.terminateEvent.is_set():
            before = datetime.now()
            before2 = time.time()

            # Read from game data shared memory
            game_data_shared_memory.seek(0)  # Move to beginning of shared memory
            ctypes.memmove(ctypes.addressof(lock), game_data_shared_memory.read(ctypes.sizeof(lock)), ctypes.sizeof(
                lock))  # dll uses InterlockedExchange so this read will return the correct value!

            if lock.value != REFRESH_IN_PROGRESS:
                game_data_shared_memory.seek(4, os.SEEK_CUR)  # Move 4 bytes past error code
                ctypes.memmove(ctypes.addressof(game_tick_packet),
                               game_data_shared_memory.read(ctypes.sizeof(gd.GameTickPacket)),
                               ctypes.sizeof(gd.GameTickPacket))  # copy shared memory into struct
            if game_tick_packet.gameInfo.bMatchEnded:
                print('\n\n\n\n Match has ended so ending bot loop\n\n\n\n\n')
                break

            controller_input = None
            # Run the Agent only if the gameInfo has updated.
            tick_game_time = game_tick_packet.gameInfo.TimeSeconds
            should_call_while_paused = datetime.now() - last_call_real_time >= MAX_AGENT_CALL_PERIOD
            if tick_game_time != last_tick_game_time or should_call_while_paused:
                last_tick_game_time = tick_game_time
                last_call_real_time = datetime.now()

                try:
                    # Reload the Agent if it has been modified.
                    new_module_modification_time = os.stat(agent_module.__file__).st_mtime
                    if new_module_modification_time != last_module_modification_time:
                        last_module_modification_time = new_module_modification_time
                        print('Reloading Agent: ' + agent_module.__file__)

                        importlib.reload(agent_module)
                        old_agent = agent
                        agent = self.load_agent(agent_module)
                        # Retire after the replacement initialized properly.
                        if hasattr(old_agent, 'retire'):
                            old_agent.retire()

                    # Call agent
                    controller_input = agent.get_output_vector(game_tick_packet)

                    if not controller_input:
                        raise Exception('Agent "{}" did not return a player_input tuple.'.format(agent_module.__file__))

                    # Write all player inputs
                    player_input.fThrottle = controller_input[0]
                    player_input.fSteer = controller_input[1]
                    player_input.fPitch = controller_input[2]
                    player_input.fYaw = controller_input[3]
                    player_input.fRoll = controller_input[4]
                    player_input.bJump = controller_input[5]
                    player_input.bBoost = controller_input[6]
                    player_input.bHandbrake = controller_input[7]

                except Exception as e:
                    traceback.print_exc()

                # Workaround for windows streams behaving weirdly when not in command prompt
                sys.stdout.flush()
                sys.stderr.flush()

            current_time = game_tick_packet.gameInfo.TimeSeconds

            if self.save_data and game_tick_packet.gameInfo.bRoundActive and not old_time == current_time and not current_time == -10:
                np_input = self.input_converter.create_input_array(game_tick_packet, passed_time=time.time() - before2)
                np_output = np.array(controller_input, dtype=np.float32)
                self.input_array = np.append(self.input_array, np_input)
                self.output_array = np.append(self.output_array, np_output)
                if self.frames % self.batch_size == 0 and not self.frames == 0:
                    print('writing big array')
                    compressor.write_array_to_file(self.game_file, self.input_array)
                    compressor.write_array_to_file(self.game_file, self.output_array)
                    self.input_array = np.array([])
                    self.output_array = np.array([])
                if self.frames % (self.batch_size * self.upload_size) == 0 and not self.frames == 0:
                    print('adding new file and uploading')
                    self.file_number += 1
                    self.game_file.close()
                    print('creating file ' + filename)
                    self.maybe_compress_and_upload(filename)
                    filename = self.create_file_name()
                    self.create_new_file(filename)
                    self.maybe_delete(self.file_number - 3)
                self.frames += 1

            old_time = current_time

            # Ratelimit here
            after = datetime.now()

            after2 = time.time()
            # cant ever drop below 50 frames
            if after2 - before2 > 0.02:
                print('Too slow for ' + self.name + ': ' + str(after2 - before2) +
                      ' frames since slowdown ' + str(counter))
                counter = 0
            else:
                counter += 1

            r.acquire(after - before)

        if hasattr(agent, 'retire'):
            agent.retire()
        # If terminated, send callback
        print("something ended closing file")
        if self.save_data:
            self.maybe_compress_and_upload(filename)
            self.server_manager.retry_files()

        print('done with bot')

        self.callbackEvent.set()

    def maybe_compress_and_upload(self, filename):
        if not os.path.isfile(filename + '.gz'):
            compressed = self.compress(filename)
            self.server_manager.maybe_upload_replay(compressed)

    def compress(self, filename):
        output = filename + '.gz'
        with open(filename, 'rb') as f_in:
            with gzip.open(output, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return output

    def maybe_delete(self, file_number):
        if file_number > 0:
            filename = self.game_name + '\\' + self.name + '-' + str(file_number) + '.bin'
            os.remove(filename)

    def create_new_file(self, filename):
        self.game_file = open(filename.replace(" ", ""), 'wb')
        compressor.write_version_info(self.game_file, compressor.get_latest_file_version())
        compressor.write_bot_hash(self.game_file, self.model_hash)
        compressor.write_is_eval(self.game_file, self.is_eval)

    def create_file_name(self):
        return self.game_name + '/' + self.name + '-' + str(self.file_number) + '.bin'
