from tkinter.filedialog import askopenfilename
import tkinter as tk
from tkinter import ttk
import os
import importlib
import inspect


class StartRunnerGUI(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        # Set parent settings
        self.parent = parent
        self.parent.title("Runner GUI")
        self.parent.iconbitmap(default="images" + os.sep + "Saltie_logo.ico")

        self.agent_frames = tk.Frame(self)
        self.agent_frames.pack_propagate(False)
        for i in [0, 1, 2]:
            self.agent_frames.grid_columnconfigure(i, weight=1, minsize=350)

        self.agent_frames.grid(row=0, column=0)
        self.agents = list()
        self.agents.append(self.AgentFrame(self.agent_frames))
        ttk.Label(self.agent_frames, text="Bot 1").grid(row=0, column=0)
        self.agents[0].grid(row=1, column=0)
        self.current_agent_frame = 0

        self.left_button = ttk.Button(self.agent_frames, text="<<",
                                      command=lambda: self.change_view(self.current_agent_frame - 1))
        self.right_button = ttk.Button(self.agent_frames, text=">>",
                                       command=lambda: self.change_view(self.current_agent_frame + 1))

        self.add_agent_frame = tk.Frame(self.agent_frames, bg='blue')
        ttk.Button(self.add_agent_frame, text="Add Agent", command=self.add_agent).grid()
        self.add_agent_frame.grid(row=1, column=1)
        ttk.Label(self.agent_frames, text="").grid(row=1, column=2)

        self.start_button = ttk.Button(self, text="Start running!", command=self.start_running, state="disabled")
        self.start_button.grid(row=1, column=0, padx=(5, 10), pady=(5, 10), sticky="se")

    def start_running(self):
        print("Gotta run that runner")

    def change_view(self, index):
        for i in [0, 1]:
            for widget in self.agent_frames.grid_slaves(row=i):
                widget.grid_forget()
        if len(self.agents) == 1:
            ttk.Label(self.agent_frames, text="Bot " + str(index + 1)).grid(row=0, column=0)
            self.agents[0].grid(row=1, column=0, sticky="nsew")
            self.add_agent_frame.grid(row=1, column=1)
            ttk.Label(self.agent_frames, text="").grid(row=1, column=2)
        elif index == len(self.agents) - 2 and index < 8:
            ttk.Label(self.agent_frames, text="Bot " + str(index + 1)).grid(row=0, column=0)
            ttk.Label(self.agent_frames, text="Bot " + str(index + 2)).grid(row=0, column=1)
            self.agents[index].grid(row=1, column=0, sticky="nsew")
            self.agents[index + 1].grid(row=1, column=1, sticky="nsew")
            self.add_agent_frame.grid(row=1, column=2)
        else:
            ttk.Label(self.agent_frames, text="Bot " + str(index + 1)).grid(row=0, column=0)
            ttk.Label(self.agent_frames, text="Bot " + str(index + 2)).grid(row=0, column=1)
            ttk.Label(self.agent_frames, text="Bot " + str(index + 3)).grid(row=0, column=2)
            self.agents[index].grid(row=1, column=0, sticky="nsew")
            self.agents[index + 1].grid(row=1, column=1, sticky="nsew")
            self.agents[index + 2].grid(row=1, column=2, sticky="nsew")
        if index < 1 and self.left_button.winfo_ismapped():
            self.left_button.grid_forget()
        elif index != 0 and not self.left_button.winfo_ismapped():
            self.left_button.grid(row=2, column=0)
        if (index > 6 or index == len(self.agents) - 2) and self.right_button.winfo_ismapped():
            self.right_button.grid_forget()
        elif not (index == 7 or index == len(self.agents) - 2) and len(self.agents) > 3 and not self.right_button.winfo_ismapped():
            self.right_button.grid(row=2, column=2)
        self.current_agent_frame = index

    def add_agent(self):
        self.agents.append(self.AgentFrame(self.agent_frames))
        if len(self.agents) == 10:
            self.change_view(len(self.agents) - 3)
        else:
            self.change_view(len(self.agents) - 2)

    def remove_agent(self, agent):
        agent.destroy()
        self.agents.remove(agent)
        if len(self.agents) == 0  :
            self.agents.append(self.AgentFrame(self.agent_frames))
        if self.current_agent_frame != 0 and self.current_agent_frame > len(self.agents) - 2:
            self.change_view(self.current_agent_frame - 1)
        else:
            self.change_view(self.current_agent_frame)

    class AgentFrame(tk.Frame):
        def __init__(self, parent, *args, **kwargs):
            tk.Frame.__init__(self, parent, *args, *kwargs)
            self.config(relief="flat", borderwidth=4)
            self.agent_path = tk.StringVar()
            self.config_options_path = {}
            self.looks_path = tk.StringVar()

            # Looks config
            self.looks_widgets = list()  # row 0
            self.looks_widgets.append(ttk.Label(self, text="Loadout path:", anchor="e"))
            self.looks_widgets.append(ttk.Entry(self, textvariable=self.looks_path, state="readonly", takefocus=False))
            self.looks_widgets.append(ttk.Button(self, text="Select file", command=self.change_looks_path))

            # rlbot.cfg options
            self.is_bot = tk.BooleanVar()  # row 1
            self.is_bot_widgets = list()
            self.is_bot_widgets.append(ttk.Label(self, text="Is bot: ", anchor="e"))
            self.is_bot_widgets.append(
                ttk.Combobox(self, textvariable=self.is_bot, values=(True, False), state="readonly"))
            self.is_bot_widgets[1].bind("<<ComboboxSelected>>", self.change_is_bot)
            self.is_bot_widgets[1].current(0)

            self.rlbot_controlled = tk.BooleanVar()  # row 2
            self.rlbot_controlled_widgets = list()
            self.rlbot_controlled_widgets.append(ttk.Label(self, text="RLBot controlled: ", anchor="e"))
            self.rlbot_controlled_widgets.append(
                ttk.Combobox(self, textvariable=self.rlbot_controlled, values=(True, False), state="readonly"))
            self.rlbot_controlled_widgets[1].bind("<<ComboboxSelected>>", self.change_rlbot_controlled)
            self.rlbot_controlled_widgets[1].current(1)

            self.bot_level = tk.StringVar(value="All-Star")  # row 3
            self.bot_level_widgets = list()
            self.bot_level_widgets.append(ttk.Label(self, text="Bot level: ", anchor="e"))
            self.bot_level_widgets.append(ttk.Combobox(self, textvariable=self.bot_level, state="readonly",
                                                       values=("Rookie", "Pro", "All-Star")))
            # Agent path
            self.agent_path_widgets = list()  # row 4
            self.agent_path_widgets.append(ttk.Label(self, text="Agent path: ", anchor="e"))
            self.agent_path_widgets.append(
                ttk.Entry(self, textvariable=self.agent_path, state="readonly", takefocus=False))
            self.agent_path_widgets.append(
                ttk.Button(self, text="Select file", command=self.change_bot_path))

            # Agent config
            self.agent_config_widgets = list()  # row 5
            self.agent_config_widgets.append(ttk.Label(self, text="Bot Parameters: ", anchor="e"))
            self.agent_config_widgets.append(ttk.Combobox(self, state="readonly"))
            self.agent_config_widgets[1].bind("<<ComboboxSelected>>", self.change_config)
            self.agent_config_widgets.append(
                ttk.Button(self, text="Add config", command=self.add_config_option, state="disabled"))

            self.custom_agent_options = tk.Frame(self, borderwidth=2, relief=tk.SUNKEN)  # row 6

            ttk.Button(self, text="Remove", command=lambda: parent.master.remove_agent(self)).grid(row=7, column=2)

            self.grid_items(0, 0, self.looks_widgets, self.is_bot_widgets)
            self.change_is_bot()
            self.change_rlbot_controlled()


        def grid_items(self, start_row=0, start_index=0, *widgets):
            for row, widget_list in enumerate(widgets):
                row += start_row
                for column, widget in enumerate(widget_list):
                    column += start_index
                    widget.grid(row=row, column=column, sticky="nsew")

        def change_is_bot(self, event=None, hide=False):
            if self.is_bot.get() and not hide:
                if self.rlbot_controlled_widgets[0].winfo_ismapped():
                    return
                self.grid_items(2, 0, self.rlbot_controlled_widgets)
                self.change_rlbot_controlled()
            else:
                if not self.rlbot_controlled_widgets[0].winfo_ismapped():
                    return
                for widget in self.grid_slaves(row=2):
                    widget.grid_forget()
                self.change_rlbot_controlled(hide=True)

        def change_rlbot_controlled(self, event=None, hide=False):
            if hide:
                for i in [3, 4, 5, 6]:
                    for widget in self.grid_slaves(row=i):
                        widget.grid_forget()
                return
            if self.rlbot_controlled.get():
                if self.rlbot_controlled_widgets[0].winfo_ismapped():
                    for widget in self.grid_slaves(row=3):
                        widget.grid_forget()
                self.grid_items(4, 0, self.agent_path_widgets, self.agent_config_widgets)
                self.custom_agent_options.grid(row=6, column=0, columnspan=3, sticky="nsew")
            else:
                for i in [4, 5, 6]:
                    for widget in self.grid_slaves(row=i):
                        widget.grid_forget()
                self.grid_items(3, 0, self.bot_level_widgets)

        def change_bot_path(self):
            agent_file_path = askopenfilename(
                initialdir=os.path.dirname(os.path.realpath(__file__)),
                filetypes=[("Python File", "*.py")],
                title="Choose a file")
            if agent_file_path:
                self.agent_path.set(agent_file_path)
                self.agent_config_widgets[2]["state"] = "normal"
                self.agent_config_widgets[1]['values'] = ("custom",)
                self.agent_config_widgets[1].set("custom")
                module = self.agent_path.get().replace(os.path.dirname(os.path.realpath(__file__)).replace(os.sep, "/"),
                                                       "", 1).replace("/", ".")[1:-3]
                trainer_package = importlib.import_module(module)
                trainer_classes = [m for m in inspect.getmembers(trainer_package, inspect.isclass) if
                                   m[1].__module__ == module]
                if len(trainer_classes) > 1:
                    popup = tk.Toplevel()
                    popup.title("Choose agent class")
                    popup.transient(self)
                    popup.grab_set()
                    popup.protocol("WM_DELETE_WINDOW", lambda: None)
                    selected = tk.IntVar()
                    tk.Label(popup, text="Select the class and press continue").grid(row=0, column=0, columnspan=2,
                                                                                     padx=(10, 10), pady=(10, 5))
                    for i in range(len(trainer_classes)):
                        ttk.Radiobutton(popup, text=trainer_classes[i][0], value=i, variable=selected).grid(
                            row=i + 1, column=0, sticky="nsew", padx=(10, 0))
                    selected.set(0)

                    def chosen_class():
                        self.agent_class = trainer_classes[selected.get()]
                        popup.destroy()

                    ttk.Button(popup, text="Continue", command=chosen_class).grid(row=len(trainer_classes), column=1,
                                                                                  padx=(0, 10), pady=(0, 10))
                    self.wait_window(popup)
                else:
                    self.agent_class = trainer_classes[0]
                self.initialise_custom_config()
                self.change_config()

        def change_looks_path(self):
            config_path = askopenfilename(
                initialdir=os.path.dirname(os.path.realpath(__file__)),
                filetypes=[("Config File", "*.cfg")],
                title="Choose a file")
            if config_path:
                self.looks_path.set(config_path)

        def change_config(self, event=None):
            config_name = self.agent_config_widgets[1].get()
            if config_name == "custom":
                if not self.custom_agent_options.winfo_ismapped():
                    self.custom_agent_options.grid(row=6, column=0, columnspan=3, sticky="nsew")
            else:
                if self.custom_agent_options.winfo_ismapped():
                    self.custom_agent_options.grid_forget()

        def add_config_option(self):
            config_path = askopenfilename(
                initialdir=os.path.dirname(os.path.realpath(__file__)),
                filetypes=[("Config File", "*.cfg")],
                title="Choose a file.")
            if config_path:
                config_name = os.path.splitext(os.path.basename(os.path.realpath(config_path)))[0]
                self.config_options_path[config_name] = config_path
                self.agent_config_widgets[1]['values'] += (config_name,)
                self.agent_config_widgets[1].set(config_name)
                self.change_config()

        def initialise_custom_config(self):
            for widget in self.custom_agent_options.grid_slaves():
                widget.grid_forget()
            try:
                self.bot_config = self.agent_class[1].get_parameters_header()
            except AttributeError:
                error = "This class does not contain a config method, unable to create custom config"
                ttk.Label(self.custom_agent_options, text=error).grid()
                return

            if not self.bot_config.values:
                ttk.Label(self.custom_agent_options, text="No Bot Parameters for this agent").grid()
                return

            ttk.Label(self.custom_agent_options, text="Bot Parameters", anchor="center").grid(row=0, column=0,
                                                                                              columnspan=2)
            for parameter_index, (parameter_name, parameter) in enumerate(self.bot_config.values.items()):
                ttk.Label(self.custom_agent_options, text=parameter_name + ":", anchor='e').grid(
                    row=parameter_index + 1, column=0, sticky="ew")
                big = 20000000
                if parameter.type == int:
                    parameter.value = tk.IntVar()
                    widget = tk.Spinbox(self.custom_agent_options, textvariable=parameter.value, from_=0, to=big)
                elif parameter.type == float:
                    parameter.value = tk.DoubleVar()
                    widget = tk.Spinbox(self.custom_agent_options, textvariable=parameter.value, from_=0, to=big,
                                        increment=.01)
                elif parameter.type == bool:
                    parameter.value = tk.BooleanVar()
                    widget = ttk.Combobox(self.custom_agent_options, textvariable=parameter.value, values=(True, False),
                                          state="readonly")
                    widget.current(0) if not parameter.default else widget.current(1)
                elif parameter.type == str:
                    parameter.value = tk.StringVar()
                    widget = ttk.Entry(self.custom_agent_options, textvariable=parameter.value)
                else:
                    widget = ttk.Label("Unknown type")

                widget.grid(row=parameter_index + 1, column=1, sticky="ew")

                if parameter.default is not None and parameter.type is not bool:
                    parameter.value.set(parameter.default)
            self.custom_agent_options.grid_columnconfigure(1, weight=1)


if __name__ == '__main__':
    root = tk.Tk()
    runner = StartRunnerGUI(root)
    runner.pack(side="top", fill="both", expand=True)
    root.mainloop()
