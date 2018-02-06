def ding():
    try:
        import winsound
        duration = 1000  # millisecond
        freq = 440  # Hz
        winsound.Beep(freq, duration)
    except Exception as e:
        pass

    try:
        import os
        duration = 1  # second
        freq = 440  # Hz
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    except Exception as e:
        pass

if __name__ == '__main__':
    ding()
