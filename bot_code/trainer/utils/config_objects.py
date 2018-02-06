class ConfigObject:
    headers = None

    def __init__(self):
        self.headers = list()

    def add_header(self, header):
        self.headers.append(header)
        return header

    def add_header_name(self, header_name):
        header = self.ConfigHeader(header_name)
        self.headers.append(header)
        return header

    def get_header(self, header_name):
        for header in self.headers:
            if header.name == header_name:
                return header
        return self.add_header_name(header_name)

    class ConfigHeader:
        def __init__(self, header_name):
            self.name = header_name
            self.values = list()

        def add_value(self, name, value_type, default=None, description=""):
            self.values.append(self.ConfigValue(name, value_type, default, description))

        class ConfigValue:
            def __init__(self, name, value_type, default=None, description=""):
                self.name = name
                self.type = value_type
                self.description = description
                self.default = default
