from rlbot.botmanager.bot_helper_process import BotHelperProcess


class BaseHiveManager(BotHelperProcess):

    def __init__(self, agent_metadata_queue, quit_event):
        super().__init__(agent_metadata_queue, quit_event)

    def start(self):
        pass
