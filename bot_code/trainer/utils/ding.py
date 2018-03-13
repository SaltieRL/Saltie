import os

# Something to get the attention of the human.
# eg. after finishing a long training session.

def ding():
    try:
        import winsound
        duration = 1000  # millisecond
        freq = 440  # Hz
        winsound.Beep(freq, duration)
    except Exception as e:
        pass

    try:
        duration = 1  # second
        freq = 440  # Hz
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    except Exception as e:
        pass

def text_to_speech(text):
    os.system(
        'PowerShell -Command "Add-Type –AssemblyName System.Speech; ' +
        '''(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{}');"'''.format(text)
    )


if __name__ == '__main__':
    ding()
