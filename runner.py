import configparser
import ctypes
import io
import mmap
import msvcrt
import multiprocessing as mp
import os
import sys
import random
import time

import bot_input_struct as bi
import bot_manager
import game_data_struct as gd
import rlbot_exception

from bot_code.conversions.server_converter import ServerConverter


PARTICPANT_CONFIGURATION_HEADER = 'Participant Configuration'
PARTICPANT_BOT_KEY_PREFIX = 'participant_is_bot_'
PARTICPANT_RLBOT_KEY_PREFIX = 'participant_is_rlbot_controlled_'
PARTICPANT_CONFIG_KEY_PREFIX = 'participant_config_'
PARTICPANT_BOT_SKILL_KEY_PREFIX = 'participant_bot_skill_'
PARTICPANT_TEAM_PREFIX = 'participant_team_'
RLBOT_CONFIG_FILE = 'rlbot.cfg'
RLBOT_CONFIGURATION_HEADER = 'RLBot Configuration'
INPUT_SHARED_MEMORY_TAG = 'Local\\RLBotInput'
BOT_CONFIG_LOADOUT_HEADER = 'Participant Loadout'
BOT_CONFIG_LOADOUT_ORANGE_HEADER = 'Participant Loadout Orange'
BOT_CONFIG_MODULE_HEADER = 'Bot Location'
USER_CONFIGURATION_HEADER = 'User Info'
BOT_CONFIG_AGENT_HEADER = 'Bot Parameters'


try:
    server_manager = ServerConverter('http://saltie.tk:5000', True, True, True, username='unknown')
except ImportError:
    server_manager = ServerConverter('', False, False, False)
    print('config.py not present, cannot upload replays to collective server')
    print('Check Discord server for information')


if server_manager.error:
    server_manager.warn_server('unable to connect to server')


def get_bot_config_file_list(botCount, config):
    config_file_list = []
    for i in range(botCount):
        config_file_list.append(config.get(PARTICPANT_CONFIGURATION_HEADER, PARTICPANT_CONFIG_KEY_PREFIX + str(i)))
    return config_file_list


# Cut off at 31 characters and handle duplicates
def get_sanitized_bot_name(dict, name):
    if name not in dict:
        new_name = name[:31]  # Make sure name does not exceed 31 characters
        dict[name] = 1
    else:
        count = dict[name]
        new_name = name[:27] + "(" + str(count + 1) + ")"  # Truncate at 27 because we can have up to '(10)' appended
        dict[name] = count + 1

    return new_name


def run_agent(terminate_event, callback_event, config_file, name, team, index, module_name, game_name, save_data, server_uploader):
    bm = bot_manager.BotManager(terminate_event, callback_event, config_file, name, team,
                                index, module_name, game_name, save_data, server_uploader)
    bm.run()


def main():
    # Set up RLBot.cfg
    framework_config = configparser.RawConfigParser()
    framework_config.read(RLBOT_CONFIG_FILE)

    # Open anonymous shared memory for entire GameInputPacket and map buffer
    buff = mmap.mmap(-1, ctypes.sizeof(bi.GameInputPacket), INPUT_SHARED_MEMORY_TAG)
    gameInputPacket = bi.GameInputPacket.from_buffer(buff)

    # Determine number of participants
    num_participants = framework_config.getint(RLBOT_CONFIGURATION_HEADER, 'num_participants')

    try:
        server_manager.set_player_username(framework_config.get(USER_CONFIGURATION_HEADER, 'username'))
    except Exception as e:
        print('username not set in config', e)
        print('using default username')

    # Retrieve bot config files
    participant_configs = get_bot_config_file_list(num_participants, framework_config)

    # Create empty lists
    bot_names = []
    bot_teams = []
    bot_modules = []
    processes = []
    callbacks = []
    bot_parameter_list = []
    name_dict = dict()

    save_data = True
    save_path = os.getcwd() + '/bot_code/training/replays'
    game_name = str(int(round(time.time() * 1000))) + '-' + str(random.randint(0, 1000))
    if save_data:
        print(save_path)
        if not os.path.exists(save_path):
            print(os.path.dirname(save_path) + ' does not exist creating')
            os.makedirs(save_path)
        if not os.path.exists(save_path + '\\' + game_name):
            os.makedirs(save_path + '\\' + game_name)
        print('gameName: ' + game_name + 'in ' + save_path)

    gameInputPacket.iNumPlayers = num_participants
    server_manager.load_config()


    num_team_0 = 0
    # Set configuration values for bots and store name and team
    for i in range(num_participants):
        bot_config_path = participant_configs[i]
        sys.path.append(os.path.dirname(bot_config_path))
        bot_config = configparser.RawConfigParser()
        if server_manager.download_config:
            if 'saltie' in os.path.basename(bot_config_path):
                bot_config._read(io.StringIO(server_manager.config_response.json()['content']), 'saltie.cfg')
            else:
                bot_config.read(bot_config_path)
        else:
            bot_config.read(bot_config_path)

        team_num = framework_config.getint(PARTICPANT_CONFIGURATION_HEADER,
                                           PARTICPANT_TEAM_PREFIX + str(i))

        loadout_header = BOT_CONFIG_LOADOUT_HEADER
        if (team_num == 1 and bot_config.has_section(BOT_CONFIG_LOADOUT_ORANGE_HEADER)):
            loadout_header = BOT_CONFIG_LOADOUT_ORANGE_HEADER

        if gameInputPacket.sPlayerConfiguration[i].ucTeam == 0:
            num_team_0 += 1

        gameInputPacket.sPlayerConfiguration[i].bBot = framework_config.getboolean(PARTICPANT_CONFIGURATION_HEADER,
                                                                                   PARTICPANT_BOT_KEY_PREFIX + str(i))
        gameInputPacket.sPlayerConfiguration[i].bRLBotControlled = framework_config.getboolean(
            PARTICPANT_CONFIGURATION_HEADER,
            PARTICPANT_RLBOT_KEY_PREFIX + str(i))
        gameInputPacket.sPlayerConfiguration[i].fBotSkill = framework_config.getfloat(PARTICPANT_CONFIGURATION_HEADER,
                                                                                      PARTICPANT_BOT_SKILL_KEY_PREFIX
                                                                                      + str(i))
        gameInputPacket.sPlayerConfiguration[i].iPlayerIndex = i

        gameInputPacket.sPlayerConfiguration[i].wName = get_sanitized_bot_name(name_dict,
                                                                               bot_config.get(loadout_header, 'name'))

        gameInputPacket.sPlayerConfiguration[i].ucTeam = team_num
        gameInputPacket.sPlayerConfiguration[i].ucTeamColorID = bot_config.getint(loadout_header,
                                                                                  'team_color_id')
        gameInputPacket.sPlayerConfiguration[i].ucCustomColorID = bot_config.getint(loadout_header,
                                                                                    'custom_color_id')
        gameInputPacket.sPlayerConfiguration[i].iCarID = bot_config.getint(loadout_header, 'car_id')
        gameInputPacket.sPlayerConfiguration[i].iDecalID = bot_config.getint(loadout_header, 'decal_id')
        gameInputPacket.sPlayerConfiguration[i].iWheelsID = bot_config.getint(loadout_header, 'wheels_id')
        gameInputPacket.sPlayerConfiguration[i].iBoostID = bot_config.getint(loadout_header, 'boost_id')
        gameInputPacket.sPlayerConfiguration[i].iAntennaID = bot_config.getint(loadout_header, 'antenna_id')
        gameInputPacket.sPlayerConfiguration[i].iHatID = bot_config.getint(loadout_header, 'hat_id')
        gameInputPacket.sPlayerConfiguration[i].iPaintFinish1ID = bot_config.getint(loadout_header,
                                                                                    'paint_finish_1_id')
        gameInputPacket.sPlayerConfiguration[i].iPaintFinish2ID = bot_config.getint(loadout_header,
                                                                                    'paint_finish_2_id')
        gameInputPacket.sPlayerConfiguration[i].iEngineAudioID = bot_config.getint(loadout_header,
                                                                                   'engine_audio_id')
        gameInputPacket.sPlayerConfiguration[i].iTrailsID = bot_config.getint(loadout_header, 'trails_id')
        gameInputPacket.sPlayerConfiguration[i].iGoalExplosionID = bot_config.getint(loadout_header,
                                                                                     'goal_explosion_id')

        if bot_config.has_section(BOT_CONFIG_AGENT_HEADER):
            try:
                bot_parameter_list.append(bot_config[BOT_CONFIG_AGENT_HEADER])
            except Exception as e:
                bot_parameter_list.append(None)
                print('failed to load bot parameters')
        else:
            bot_parameter_list.append(None)


        bot_names.append(bot_config.get(loadout_header, 'name'))
        bot_teams.append(framework_config.getint(PARTICPANT_CONFIGURATION_HEADER, PARTICPANT_TEAM_PREFIX + str(i)))
        if gameInputPacket.sPlayerConfiguration[i].bRLBotControlled:
            bot_modules.append(bot_config.get(BOT_CONFIG_MODULE_HEADER, 'agent_module'))
        else:
            bot_modules.append('NO_MODULE_FOR_PARTICIPANT')
        # downloads the model based on the hash in the config
        try:
            server_manager.load_model(bot_config[BOT_CONFIG_AGENT_HEADER]['model_hash'])
        except Exception as e:
            print ("Couldn't get model hash,", e)
    server_manager.set_player_amount(num_participants, num_team_0)

    # Create Quit event
    quit_event = mp.Event()

    # Launch processes
    for i in range(num_participants):
        if gameInputPacket.sPlayerConfiguration[i].bRLBotControlled:
            callback = mp.Event()
            callbacks.append(callback)
            process = mp.Process(target=run_agent,
                                 args=(quit_event, callback, bot_parameter_list[i],
                                       str(gameInputPacket.sPlayerConfiguration[i].wName),
                                       bot_teams[i], i, bot_modules[i], save_path + '\\' + game_name,
                                       save_data, server_manager))
            process.start()

    print("Successfully configured bots. Setting flag for injected dll.")
    gameInputPacket.bStartMatch = True

    # Wait 100 milliseconds then check for an error code
    time.sleep(0.1)
    game_data_shared_memory = mmap.mmap(-1, ctypes.sizeof(gd.GameTickPacketWithLock),
                                        bot_manager.OUTPUT_SHARED_MEMORY_TAG)
    bot_output = gd.GameTickPacketWithLock.from_buffer(game_data_shared_memory)
    if not bot_output.iLastError == 0:
        # Terminate all process and then raise an exception
        quit_event.set()
        terminated = False
        while not terminated:
            terminated = True
            for callback in callbacks:
                if not callback.is_set():
                    terminated = False
        raise rlbot_exception.RLBotException().raise_exception_from_error_code(bot_output.iLastError)

    print("Press any character to exit")
    msvcrt.getch()

    print("Shutting Down")
    quit_event.set()
    # Wait for all processes to terminate before terminating main process
    terminated = False
    while not terminated:
        terminated = True
        for callback in callbacks:
            if not callback.is_set():
                terminated = False


if __name__ == '__main__':
    main()
