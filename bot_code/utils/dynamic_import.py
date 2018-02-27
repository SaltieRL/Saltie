import importlib
import inspect

'''
Imports things (classes, variables) from within the bot_code directory.
Takes strings to the given things.
'''

def get_class(package, class_name):
    return get_field(package, class_name, predicate=inspect.isclass)

def get_field(package, field_name, predicate=None):
    package = importlib.import_module('bot_code.' + package)
    module_classes = inspect.getmembers(package, predicate)
    for class_group in module_classes:
        if class_group[0] == field_name:
            return class_group[1]
    return None
