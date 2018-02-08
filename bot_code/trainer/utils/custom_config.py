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
        return self.headers[section].values[option].value.get()

    def getint(self, section, option):
        return int(self.headers[section].values[option].value.get())

    def getboolean(self, section, option):
        return bool(self.headers[section].values[option].value.get())

    def getfloat(self, section, option):
        return float(self.headers[section].values[option].value.get())

    class ConfigHeader:
        def __init__(self):
            self.values = {}

        def __getitem__(self, x):
            return self.values[x]

        def add_value(self, name, value_type, default=None, description="", var=None):
            self.values[name] = self.ConfigValue(value_type, default=default, description=description, var=var)

        def get(self, option):
            return self.values[option].value.get()

        def getint(self, option):
            return int(self.values[option].value.get())

        def getboolean(self, option):
            return bool(self.values[option].value.get())

        def getfloat(self, option):
            return float(self.values[option].value.get())

        class ConfigValue:
            def __init__(self, value_type, default=None, description="", var=None):
                self.type = value_type
                self.value = var
                self.default = default
                self.description = description
