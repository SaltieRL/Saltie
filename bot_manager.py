import ctypes
from datetime import datetime
import gzip
import hashlib
import importlib
import mmap
import os
import shutil
import time

import numpy as np

import bot_input_struct as bi
import game_data_struct as gd
import rate_limiter


from conversions import input_formatter, binary_converter as compressor


OUTPUT_SHARED_MEMORY_TAG = 'Local\\RLBotOutput'
INPUT_SHARED_MEMORY_TAG = 'Local\\RLBotInput'
RATE_LIMITED_ACTIONS_PER_SECOND = 60
REFRESH_IN_PROGRESS = 1
REFRESH_NOT_IN_PROGRESS = 0
MAX_CARS = 10


class BotManager:

    game_file = None
    model_hash = None
    is_eval = False

    def __init__(self, terminateEvent, callbackEvent, config_file, name, team, index, modulename, gamename, savedata, server_manager):
        self.terminateEvent = terminateEvent
        self.callbackEvent = callbackEvent
        self.name = name
        self.team = team
        self.index = index
        self.save_data = savedata
        self.module_name = modulename
        self.game_name = gamename
        self.input_converter = input_formatter.InputFormatter(team, index)
        self.frames = 0
        self.file_number = 1
        self.config_file = config_file
        self.server_manager = server_manager
        self.input_array = np.array([])
        self.output_array = np.array([])
        self.batch_size = 1000
        self.upload_size = 20

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

        # Get bot module
        agent_module = importlib.import_module(self.module_name)

        # Create Ratelimiter
        r = rate_limiter.RateLimiter(RATE_LIMITED_ACTIONS_PER_SECOND)

        # Find car with same name and assign index
        for i in range(MAX_CARS):
            if str(bot_output.gamecars[i].wName) == self.name:
                self.index = i
                continue

        # Create bot from module
        try:
            agent = agent_module.Agent(self.name, self.team, self.index, config_file=self.config_file)
        except TypeError:
            agent = agent_module.Agent(self.name, self.team, self.index)

        if hasattr(agent, 'create_model_hash'):
            self.model_hash = agent.create_model_hash()
        else:
            self.model_hash = int(hashlib.sha256(self.name.encode('utf-8')).hexdigest(), 16) % 2 ** 64

        self.server_manager.set_model_hash(self.model_hash)

        if hasattr(agent, 'is_evaluating'):
            self.is_eval = agent.is_evaluating
            self.server_manager.set_is_eval(self.is_eval)

        if self.save_data:
            filename = self.game_name + '\\' + self.name + '-' + str(self.file_number) + '.bin'
            print('creating file ' + filename)
            self.create_new_file(filename)
        old_time = 0
        counter = 0

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

            # Call agent
            controller_input = agent.get_output_vector(game_tick_packet)

            # Write all player inputs
            player_input.fThrottle = controller_input[0]
            player_input.fSteer = controller_input[1]
            player_input.fPitch = controller_input[2]
            player_input.fYaw = controller_input[3]
            player_input.fRoll = controller_input[4]
            player_input.bJump = controller_input[5]
            player_input.bBoost = controller_input[6]
            player_input.bHandbrake = controller_input[7]

            current_time = game_tick_packet.gameInfo.TimeSeconds

            if self.save_data and game_tick_packet.gameInfo.bRoundActive and not old_time == current_time and not current_time == -10:
                np_input, _ = self.input_converter.create_input_array(game_tick_packet)
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
                    filename = self.game_name + '\\' + self.name + '-' + str(self.file_number) + '.bin'
                    self.create_new_file(filename)
                    self.maybe_delete(self.file_number - 3)
                self.frames += 1

            old_time = current_time

            # Ratelimit here
            after = datetime.now()
            after2 = time.time()
            # cant ever drop below 30 frames
            if after2 - before2 > 0.02:
                print('Too slow for ' + self.name + ': ' + str(after2 - before2) +
                      ' frames since slowdown ' + str(counter))
                counter = 0
            else:
                counter += 1

            r.acquire(after - before)

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
