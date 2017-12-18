import live_data_util

# In order to run this you need to install pyqtgraph and pyqt4 (haven't tried other versions)
# Run python -m pyqtgraph.examples for example code
# https://github.com/pyqtgraph/pyqtgraph
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="RLBot Live Plotting")
win.resize(500,300)
win.setWindowTitle('RLBot Live Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

# Open shared memory rotating buffer
rb = live_data_util.RotatingBuffer(0)
rb2 = live_data_util.RotatingBuffer(1)

p6 = win.addPlot(title="RLBot Expected Rewards")
blue_curve = p6.plot(pen='b')
orng_curve = p6.plot(pen='y')
def update():
    global blue_curve, orng_curve, p6
    blue_curve.setData(rb.get_current_buffer())
    orng_curve.setData(rb2.get_current_buffer())
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
