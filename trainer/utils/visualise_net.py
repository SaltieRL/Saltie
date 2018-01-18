from tkinter import *
import numpy as np

above_white = 10
left_white = 10
highrelu = 10
x_spacing = 100
y_spacing = 50
circle_dia = 30


class AutoScrollbar(Scrollbar):
    ''' A scrollbar that hides itself if it's not needed.
        Works only if you use the grid geometry manager '''
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        Scrollbar.set(self, lo, hi)

class Visualiser:
    def __init__(self):
        self.gui = Tk()
        self.gui.geometry('600x450+300+100')
        self.gui.title("Net visualisation")

        # Are activations relu:
        self.relu = [True, True, True, True, False]
        self.output = np.array([[0, 1, 3, 4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 0], [5, 2, 43, 34, 234, 3, 4], [5, 7, 2, 5, 7, 19], [1, 0, 0.5, 0.1, 0.6]])
        self.biggestarraylen = 0
        self.centres = []
        self.last_layer = list()
        for item in self.output:
            if len(item) > self.biggestarraylen:
                self.biggestarraylen = len(item)
        self.eFrame = Frame(self.gui)
        self.edit_stuff()
        self.eFrame.grid(row=0, column=0)

        self.cFrame = Frame(self.gui, width=600, height=450)
        self.canvas_stuff()
        self.cFrame.grid(row=0, column=1, sticky='w')
        mainloop()

    def create_circle(self, x0, y0, activation, relu):
        if relu:
            activation = activation if activation <= highrelu else highrelu
            rgb = int(-1 * (activation - highrelu) * 25.5)
        else:
            rgb = int(-1 * (activation - 1) * 255)
        hexcolor = "#{:02x}{:02x}{:02x}".format(rgb, rgb, rgb)
        self.canvas.create_oval(x0, y0, x0 + circle_dia, y0 + circle_dia, fill=hexcolor, tags='circle')

    def create_line(self, x0, y0, x1, y1):
        half = .5 * circle_dia
        self.canvas.create_line(x0 + half, y0 + half, x1 + half, y1 + half, tags='line')

    def create_layer(self, layer):
        activations = self.output[layer]
        relu = self.relu[layer]
        x = layer * x_spacing + left_white
        y = (self.biggestarraylen - len(activations)) * y_spacing * .5 + above_white
        this_layer = list()
        for i in activations:
            this_layer.append([x, y])
            if layer != 0:
                for n in self.last_layer:
                    self.create_line(n[0], n[1], x, y)
            self.create_circle(x, y, i, relu)
            y += y_spacing
        self.last_layer = this_layer

    def edit_stuff(self):
        print("Test")

    def canvas_stuff(self):
        vbar = AutoScrollbar(self.cFrame, orient='vertical')
        hbar = AutoScrollbar(self.cFrame, orient='horizontal')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='we')

        max_width = left_white + (len(self.output) - 1) * x_spacing + circle_dia + left_white
        max_height = above_white + (self.biggestarraylen - 1) * y_spacing + circle_dia + above_white
        self.canvas = Canvas(self.cFrame, width=550, height=400, xscrollcommand=hbar.set,
                             yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0)
        for i in range(len(self.output)):
            self.create_layer(i)
        self.canvas.tag_lower('line')

        vbar.configure(command=self.canvas.yview)  # bind scrollbars to the canvas
        hbar.configure(command=self.canvas.xview)
        # Make the canvas expandable
        self.cFrame.rowconfigure(0, weight=1)
        self.cFrame.columnconfigure(0, weight=1)
        # Bind events to the Canvas
        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>', self.move_to)
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.imscale = 1.0
        self.imageid = None
        self.delta = 0.75
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def wheel(self, event):
        ''' Zoom with mouse wheel '''
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:
            scale *= self.delta
            self.imscale *= self.delta
        if event.num == 4 or event.delta == 120:
            scale /= self.delta
            self.imscale /= self.delta
        # Rescale all canvas objects
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.canvas.scale('all', x, y, scale, scale)
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))


