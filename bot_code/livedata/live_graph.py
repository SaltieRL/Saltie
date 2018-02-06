from bot_code.livedata import live_data_util

# In order to run this you need to install pyqtgraph and pyqt4 (haven't tried other versions)
# Run python -m pyqtgraph.examples for example code
# https://github.com/pyqtgraph/pyqtgraph
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg


def create_plot(name):
    p6 = win.addPlot(title=name)
    p6.setYRange(-1.5, 1.5)
    blue_curve = p6.plot(pen='g')
    orng_curve = p6.plot(pen='r')
    #green_curve = p6.plot(pen='g')
    #red_curve = p6.plot(pen='r')
    return p6, blue_curve, orng_curve

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="RLBot Live Plotting")
win.resize(500, 300)
win.setWindowTitle('RLBot Live Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

# Open shared memory rotating buffer
expected_reward_blue_0 = live_data_util.RotatingBuffer(0)
expected_reward_orange_0 = live_data_util.RotatingBuffer(1)
real_reward_blue_0 = live_data_util.RotatingBuffer(10)
real_reward_orange_0 = live_data_util.RotatingBuffer(11)

realPlot, real_blue, real_orange = create_plot("RLBot Real Rewards")
expectedPlot, expected_blue, expected_orange = create_plot("RLBot Expected Rewards")
def update():
    global realPlot, real_blue, real_orange, expectedPlot, expected_blue, expected_orange
    expected_blue.setData(expected_reward_blue_0.get_current_buffer())
    expected_orange.setData(expected_reward_orange_0.get_current_buffer())
    real_blue.setData(real_reward_blue_0.get_current_buffer())
    real_orange.setData(real_reward_orange_0.get_current_buffer())

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
