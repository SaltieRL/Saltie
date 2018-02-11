import tkinter as tk


class ConfigObject:
    def __init__(self):
        self.headers = {}

    def __getitem__(self, x):
        return self.get_header(x)

    def add_header(self, header_name, header):
        self.headers[header_name] = header
        return header

    def add_header_name(self, header_name):
        header = self.ConfigHeader()
        self.headers[header_name] = header
        return header

    def get_header(self, header_name):
        if header_name in self.headers:
            return self.headers[header_name]
        return self.add_header_name(header_name)

    def get(self, section, option):
        return self[section][option].value.get() if isinstance(self[section][option].value, tk.StringVar) \
            else self[section][option].value

    def getint(self, section, option):
        return self[section][option].value.get() if isinstance(self[section][option].value, tk.IntVar) \
            else int(self[section][option].value)

    def getboolean(self, section, option):
        return self[section][option].value.get() if isinstance(self[section][option].value, tk.BooleanVar) \
            else bool(self[section][option].value)

    def getfloat(self, section, option):
        return float(self[section][option].value.get()) if isinstance(self[section][option].value, tk.DoubleVar) \
            else float(self[section][option].value)

    class ConfigHeader:
        def __init__(self):
            self.values = {}

        def __getitem__(self, x):
            return self.values[x]

        def add_value(self, name, value_type, default=None, description="", var=None):
            self.values[name] = self.ConfigValue(value_type, default=default, description=description, var=var)

        def get(self, option):
            return self[option].value.get() if isinstance(self[option].value, tk.StringVar) \
                else self[option].value

        def getint(self, option):
            return self[option].value.get() if isinstance(self[option].value, tk.IntVar) \
                else int(self[option].value)

        def getboolean(self, option):
            return self[option].value.get() if isinstance(self[option].value, tk.BooleanVar) \
                else bool(self[option].value)

        def getfloat(self, option):
            return float(self[option].value.get()) if isinstance(self[option].value, tk.DoubleVar) \
                else float(self[option].value)

        class ConfigValue:
            def __init__(self, value_type, default=None, description="", var=None):
                self.type = value_type
                self.value = var
                self.default = default
                self.description = description
