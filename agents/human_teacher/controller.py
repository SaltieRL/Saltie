from pynput.keyboard import Listener, Key
from rlbot.agents.base_agent import SimpleControllerState


def deadzone(normalized_axis):
    if abs(normalized_axis) < 0.1:
        return 0.0
    return normalized_axis


class HytakControllerInput(SimpleControllerState):
    def __init__(self):
        self._gas_pedal = 0.0
        self._brake_pedal = 0.0

        self._left = 0.0
        self._right = 0.0

        self._up = 0.0
        self._down = 0.0

        self._roll_left = 0.0
        self._roll_right = 0.0

        self.jump = False
        self.boost = False
        self.handbrake = False

        self.listener = Listener(self.create_on_press(), self.create_on_release())
        self.listener.start()

    @property
    def throttle(self):
        return self._gas_pedal - self._brake_pedal

    @property
    def steer(self):
        return self._right - self._left

    @property
    def yaw(self):
        return self.steer

    @property
    def pitch(self):
        return self._up - self._down

    @property
    def roll(self):
        return self._roll_right - self._roll_left

    def create_on_press(self):
        def on_press(key):
            if key is None:
                return
            elif isinstance(key, Key):
                if key == Key.space:
                    self.jump = True
            elif key.char == '8':
                self._gas_pedal = 1.0
            elif key.char == '5':
                self._brake_pedal = 1.0
            elif key.char == 'a':
                self._left = 1.0
            elif key.char == 'd':
                self._right = 1.0
            elif key.char == 's':
                self._up = 1.0
            elif key.char == 'w':
                self._down = 1.0
            elif key.char == '0':
                self.boost = True
            elif key.char == '4':
                self.handbrake = True
            elif key.char == 'q':
                self._roll_left = 1.0
            elif key.char == 'e':
                self._roll_right = 1.0
        return on_press

    def create_on_release(self):
        def on_release(key):
            if key is None:
                return
            elif isinstance(key, Key):
                if key == Key.space:
                    self.jump = False
            elif key.char == '8':
                self._gas_pedal = 0.0
            elif key.char == '5':
                self._brake_pedal = 0.0
            elif key.char == 'a':
                self._left = 0.0
            elif key.char == 'd':
                self._right = 0.0
            elif key.char == 's':
                self._up = 0.0
            elif key.char == 'w':
                self._down = 0.0
            elif key.char == '0':
                self.boost = False
            elif key.char == '4':
                self.handbrake = False
            elif key.char == 'q':
                self._roll_left = 0.0
            elif key.char == 'e':
                self._roll_right = 0.0
        return on_release

    def __eq__(self, other):
        return self.roll == other.roll and\
            self.jump == other.jump and\
            self.boost == other.boost and\
            self.handbrake == other.handbrake and\
            self.throttle == other.throttle and\
            self.steer == other.steer and\
            self.yaw == other.yaw and\
            self.pitch == other.pitch
