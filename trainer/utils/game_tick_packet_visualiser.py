# -*- coding: utf-8 -*-
"""
    Animated 3D sinc function
"""
'''
Install pyopengl with "pip install PyOpenGL PyOpenGL_accelerate"
Install PyQt5 with "pip install PyQt5"
Install pyqtgraph with "pip install pyqtgraph"
'''
import os

os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys
import mmap
import ctypes
import game_data_struct as gds
import math

CAR_WIDTH = 300.0
UCONST_Pi = 3.1415926
URotation180 = float(32768)
URotationToRadians = UCONST_Pi / URotation180

class Visualizer(object):
    def __init__(self, game_tick_packet):
        self.traces = dict()

        self.app = pg.Qt.QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 40
        self.w.setWindowTitle('pyqtgraph example: GLLinePlotItem')
        self.w.setGeometry(0, 50, 1920, 1080)
        self.w.show()
        # Open anonymous shared memory for entire GameInputPacket

        # Map buffer to ctypes structure
        self.game_tick_packet = game_tick_packet

        # create the background grids
        gx = gl.GLGridItem()
        gx.setSize(12000, 12000, 10000)
        gx.rotate(90, 0, 1, 0)
        gx.translate(-6000, 0, 0)
        gx.setSpacing(100, 100, 100)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -6000, 0)
        gy.setSize(12000, 12000, 10000)
        gy.setSpacing(100, 100, 100)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.setSize(12000, 12000, 10000)
        gz.setSpacing(100, 100, 100)
        self.w.addItem(gz)


        # Adding rlbot cars
        bluecar_x = self.game_tick_packet.gamecars[0].Location.X
        bluecar_y = self.game_tick_packet.gamecars[0].Location.Y
        bluecar_z = self.game_tick_packet.gamecars[0].Location.Z
        bluecar_x_plot = np.array([bluecar_x, bluecar_x + CAR_WIDTH])
        bluecar_y_plot = np.array([bluecar_y, bluecar_y + CAR_WIDTH])
        bluecar_z_plot = np.array([bluecar_z, bluecar_z + CAR_WIDTH])
        bluepts = np.vstack([bluecar_x_plot, bluecar_y_plot, bluecar_z_plot]).transpose()
        someitem = gl.GLScatterPlotItem(pos=bluepts, color=pg.glColor('b'), width=CAR_WIDTH, antialias=True)
        self.bluecar = gl.GLLinePlotItem(pos=bluepts, color=pg.glColor('b'), width=CAR_WIDTH, antialias=True)
        self.w.addItem(self.bluecar)

        orngcar_x = self.game_tick_packet.gamecars[1].Location.X
        orngcar_y = self.game_tick_packet.gamecars[1].Location.Y
        orngcar_z = self.game_tick_packet.gamecars[1].Location.Z
        orngcar_x_plot = np.array([orngcar_y, orngcar_y + CAR_WIDTH])
        orngcar_y_plot = np.array([orngcar_z, orngcar_z + CAR_WIDTH])
        orngcar_z_plot = np.array([-1.0 * orngcar_x, -1.0 * orngcar_x + CAR_WIDTH])
        orngpts = np.vstack([orngcar_x_plot, orngcar_y_plot, orngcar_z_plot]).transpose()
        self.orngcar = gl.GLLinePlotItem(pos=orngpts, color=pg.glColor('r'), width=CAR_WIDTH, antialias=True)
        self.w.addItem(self.orngcar)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(pg.Qt.QtCore, 'PYQT_VERSION'):
            pg.Qt.QtGui.QApplication.instance().exec_()

    def set_plot_car_data(self):
        bluecar_x = self.game_tick_packet.gamecars[0].Location.X
        bluecar_y = self.game_tick_packet.gamecars[0].Location.Y
        bluecar_z = self.game_tick_packet.gamecars[0].Location.Z

        bluecar_pitch = self.game_tick_packet.gamecars[0].Rotation.Pitch
        bluecar_yaw = self.game_tick_packet.gamecars[0].Rotation.Yaw
        # Nose vector x component
        bluecar_nose_vect_x = math.cos(bluecar_pitch * URotationToRadians) * math.cos(bluecar_yaw * URotationToRadians)
        # Nose vector y component
        bluecar_nose_vect_y = math.cos(bluecar_pitch * URotationToRadians) * math.sin(bluecar_yaw * URotationToRadians)
        # Nose vector z component
        bluecar_nose_vect_z = math.sin(bluecar_pitch * URotationToRadians)

        # Converting from Unreal Engine coordinate system to graphing coordinate system (I think it's the same actually)
        bluecar_x_plot = np.array([bluecar_x, bluecar_x + bluecar_nose_vect_x * CAR_WIDTH])
        bluecar_y_plot = np.array([bluecar_y, bluecar_y + bluecar_nose_vect_y * CAR_WIDTH])
        bluecar_z_plot = np.array([bluecar_z, bluecar_z + bluecar_nose_vect_z * CAR_WIDTH])

        bluepts = np.vstack([bluecar_y_plot, bluecar_x_plot, bluecar_z_plot]).transpose()
        self.bluecar.setData(pos=bluepts, color=pg.glColor('b'), width=CAR_WIDTH)

        orngcar_x = self.game_tick_packet.gamecars[1].Location.X
        orngcar_y = self.game_tick_packet.gamecars[1].Location.Y
        orngcar_z = self.game_tick_packet.gamecars[1].Location.Z

        orngcar_pitch = self.game_tick_packet.gamecars[1].Rotation.Pitch
        orngcar_yaw = self.game_tick_packet.gamecars[1].Rotation.Yaw
        # Nose vector x component
        orngcar_nose_vect_x = math.cos(orngcar_pitch * URotationToRadians) * math.cos(orngcar_yaw * URotationToRadians)
        # Nose vector y component
        orngcar_nose_vect_y = math.cos(orngcar_pitch * URotationToRadians) * math.sin(orngcar_yaw * URotationToRadians)
        # Nose vector z component
        orngcar_nose_vect_z = math.sin(orngcar_pitch * URotationToRadians)

        # Converting from Unreal Engine coordinate system to graphing coordinate system (I think it's the same actually)
        orngcar_x_plot = np.array([orngcar_x, orngcar_x + orngcar_nose_vect_x * CAR_WIDTH])
        orngcar_y_plot = np.array([orngcar_y, orngcar_y + orngcar_nose_vect_y * CAR_WIDTH])
        orngcar_z_plot = np.array([orngcar_z, orngcar_z + orngcar_nose_vect_z * CAR_WIDTH])

        orngpts = np.vstack([orngcar_y_plot, orngcar_x_plot, orngcar_z_plot]).transpose()
        self.orngcar.setData(pos=orngpts, color=pg.glColor('r'), width=CAR_WIDTH)

    def update(self):
        self.set_plot_car_data()

    def animation(self):
        timer = pg.Qt.QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        self.start()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    v = Visualizer()
    v.animation()
