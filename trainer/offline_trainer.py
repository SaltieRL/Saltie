import os
import time
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from conversions import binary_converter
from trainer import reward_trainer
from trainer.model_eval_trainer import EvalTrainer


def get_all_files():
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    training_path = dir_path + '\\training'
    files = []
    exclude_paths = {'data', 'ignore'}
    exclude_files = {''}
    for (dirpath, dirnames, filenames) in os.walk(training_path):
        dirnames[:] = [d for d in dirnames if d not in exclude_paths]
        for file in filenames:
            skip_file = False
            for excluded_name in exclude_files:
                if excluded_name in file and excluded_name != '':
                    print('exclude file: ' + file)
                    skip_file = True
                    break
            if skip_file:
                continue
            files.append(dirpath + '\\' + file)
    return files


def get_trainer_class():
    # fill your input function here!
    return EvalTrainer


if __name__ == '__main__':
    files = get_all_files()
    print('training on ' + str(len(files)) + ' files')
    trainerClass = get_trainer_class()()
    counter = 0
    total_time = 0
    try:
        for file in files:
            start = time.time()
            counter += 1
            try:
                with open(file, 'r+b') as f:
                    trainerClass.start_new_file()
                    print('running file ' + file + 'file ' + str(counter) + '/' + str(len(files)))
                    binary_converter.read_data(f, trainerClass.process_pair)
                    trainerClass.end_file()
                end = time.time()
                difference = end - start
                total_time += difference
                print('trained file in ' + str(difference) + 's')
            except FileNotFoundError:
                print('whoops file not found')
                print(file)
    finally:
        print('ran through all files in ' + str(total_time / 60) + 'm')
        print('average time: ' + str((total_time / len(files))) + 's')
        trainerClass.end_everything()
