import os
import random


def get_file_get_function(download, input_server):
    if download:
        return input_server.download_file
    else:
        return lambda input_file: input_file

def get_file_list_get_function(download, input_server):
    if download:
        return input_server.get_replays
    else:
        return get_all_files

def get_all_files(max_files, only_eval):
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    training_path = os.path.join(dir_path,'training', 'replays')
    files = []
    include_extensions = {'gz'}
    exclude_paths = {'data', 'ignore'}
    exclude_files = {''}
    for (dirpath, dirnames, filenames) in os.walk(training_path):
        dirnames[:] = [d for d in dirnames if d not in exclude_paths]
        for file in filenames:
            skip_file = False
            if file.split('.')[-1] not in include_extensions:
                continue
            for excluded_name in exclude_files:
                if excluded_name in file and excluded_name != '':
                    print('exclude file: ' + file)
                    skip_file = True
                    break
            if skip_file:
                continue
            files.append(os.path.join(dirpath, file))
    random.shuffle(files)
    return files[:max_files]
