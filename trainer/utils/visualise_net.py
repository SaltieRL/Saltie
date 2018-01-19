from tkinter import *
import numpy as np

above_white = 20
left_white = 20
highrelu = 20
x_spacing = 100
y_spacing = 50
circle_dia = 30
invert_canvas = True

class AutoScrollbar(Scrollbar):
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        Scrollbar.set(self, lo, hi)

class Visualiser:
    def __init__(self):
        self.gui = Tk()
        self.gui.geometry('600x600')
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

        self.rFrame = Frame(self.gui, width=200, height=450)
        self.rFrame.grid(row=0, column=0)

        self.eFrame = Frame(self.rFrame, width=200, height=250)
        self.edit_stuff()
        self.eFrame.grid(row=0, column=0)

        self.iFrame = Frame(self.rFrame, width=200, height=200)
        self.info_stuff()
        self.iFrame.grid(row=1, column=0)

        self.cFrame = Frame(self.gui, width=400, height=450)
        self.canvas_stuff()
        self.cFrame.grid(row=0, column=1, sticky='w')
        mainloop()

    def edit_stuff(self):
        self.relu_number = Spinbox(self.eFrame, from_=1, to=100)
        self.relu_number.grid(row=1, column=0)
        Label(self.eFrame, text="Customisation part is still wip").grid(row=0, column=0)

    def info_stuff(self):
        self.info_activation = StringVar()
        activation_label = Label(self.iFrame, textvariable=self.info_activation)
        activation_label.grid(row=0, column=0)

    def info_update_neuron(self, layer, neuron):
        self.info_activation.set("Layer:" + str(layer) + "\n" + "Neuron:" + str(neuron))

    def create_circle(self, x0, y0, activation, relu, layer, neuron):
        if invert_canvas:
            x0, y0 = y0, x0
        if relu:
            activation = activation if activation <= highrelu else highrelu
            rgb = int(-1 * (activation - highrelu) * 255 / highrelu)
        else:
            rgb = int(-1 * (activation - 1) * 255)
        hexcolor = "#{:02x}{:02x}{:02x}".format(rgb, rgb, rgb)
        tag = str(layer) + ";" + str(neuron)
        self.canvas.create_oval(x0, y0, x0 + circle_dia, y0 + circle_dia, fill=hexcolor, tags=tag)

        def handler(event, l=layer, n=neuron):
            self.info_update_neuron(l, n)
        self.canvas.tag_bind(tag, "<Motion>", handler)

    def create_line(self, x0, y0, x1, y1):
        if invert_canvas:
            x0, y0, x1, y1 = y0, x0, y1, x1
        half = .5 * circle_dia
        self.canvas.create_line(x0 + half, y0 + half, x1 + half, y1 + half, tags='line')

    def create_layer(self, layer):
        activations = self.output[layer]
        relu = self.relu[layer]
        x = layer * x_spacing + left_white
        y = (self.biggestarraylen - len(activations)) * y_spacing * .5 + above_white
        this_layer = list()
        neuron = 0
        for i in activations:
            this_layer.append([x, y])
            if layer != 0:
                for n in self.last_layer:
                    self.create_line(n[0], n[1], x, y)
            self.create_circle(x, y, i, relu, layer, neuron)
            y += y_spacing
            neuron += 1
        self.last_layer = this_layer

    def canvas_stuff(self):
        vbar = AutoScrollbar(self.cFrame, orient='vertical')
        hbar = AutoScrollbar(self.cFrame, orient='horizontal')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='we')

        self.canvas = Canvas(self.cFrame, width=400, height=450, xscrollcommand=hbar.set,
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
        self.canvas.bind('<Button-1>', self.move_from)
        self.canvas.bind('<B1-Motion>', self.move_to)
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.imscale = 1.0
        self.imageid = None
        self.delta = 0.75
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def move_from(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def wheel(self, event):
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


