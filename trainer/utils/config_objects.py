class Config:
    headers = None

    def __init__(self):
        self.headers = list()

    def add_header(self, header):
        self.headers.append(header)

    def add_header_name(self, header_name):
        self.headers.append(self.ConfigHeader(header_name))

    class ConfigHeader:
        def __init__(self, header_name):
            self.name = header_name
            self.values = list()

        def add_value(self, name, value_type, description=""):
            self.values.append(self.ConfigValue(name, value_type, description))

        class ConfigValue:
            def __init__(self, name, value_type, description=""):
                self.name = name
                self.type = value_type
                self.description = description
