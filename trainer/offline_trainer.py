import os
from conversions import binary_converter
from trainer import nnatba_trainer

def get_all_files():
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    training_path = dir_path + '\\training'
    files = []
    for (dirpath, dirnames, filenames) in os.walk(training_path):
        for file in filenames:
            files.append(dirpath + '\\' + file)
    return files


def get_trainer_class():
    #fill your input function here!
    return nnatba_trainer.NNAtbaTrainer


if __name__ == '__main__':
    files = get_all_files()
    print('training on files')
    print(files)
    trainerClass = get_trainer_class()()
    for file in files:
        with open(file, 'r+b') as f:
            trainerClass.start_new_file()
            print('running file ' + file)
            binary_converter.read_data(f, trainerClass.process_pair)
            trainerClass.end_file()
