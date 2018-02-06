class CConfig:
    def __init__(self, values):
        self.values = values

    def get(self, section, option):
        return self.values[section][option].get()

    def getint(self, section, option):
        return int(self.values[section][option].get())

    def getboolean(self, section, option):
        return bool(self.values[section][option].get())

    def getfloat(self, section, option):
        return float(self.values[section][option].get())
