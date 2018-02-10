from tkinter.filedialog import askopenfilename
import tkinter as tk
from tkinter import ttk
import os
import importlib
import inspect


class AutoScrollbar(ttk.Scrollbar):
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        ttk.Scrollbar.set(self, lo, hi)

class StartRunnerGUI(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        # Set parent settings
        self.parent = parent
        self.parent.title("Runner GUI")
        self.parent.iconbitmap(default="images" + os.sep + "Saltie_logo.ico")

        self.agent_frames = tk.Frame(self)

        self.agent_frames.grid(row=0, column=0)
        self.agents = list()
        self.agents.append(self.AgentFrame(self.agent_frames))
        ttk.Label(self.agent_frames, text="Bot 1").grid(row=0, column=0)
        self.agents[0].grid(row=1, column=0)
        self.current_agent_frame = 0

        self.left_button = ttk.Button(self.agent_frames, text="<<", command=lambda: self.change_view(self.current_agent_frame - 1))
        self.right_button = ttk.Button(self.agent_frames, text=">>", command=lambda: self.change_view(self.current_agent_frame + 1))

        self.add_agent_frame = tk.Frame(self.agent_frames)
        ttk.Button(self.add_agent_frame, text="Add Agent", command=self.add_agent).pack()
        self.add_agent_frame.grid(row=1, column=1)

        self.start_button = ttk.Button(self, text="Start running!", command=self.start_running, state="disabled")
        self.start_button.grid(row=1, column=1, padx=(5, 10), pady=(5, 10), sticky="sw")

    def start_running(self):
        print("Gotta run that runner")

    def add_agent(self):
        self.agents.append(self.AgentFrame(self.agent_frames))
        if len(self.agents) == 10:
            self.change_view(len(self.agents) - 3)
        else:
            self.change_view(len(self.agents) - 2)

    def change_view(self, index):
        for i in [0, 1]:
            for widget in self.agent_frames.grid_slaves(row=i):
                widget.grid_forget()
        if len(self.agents) == 1:
            ttk.Label(self.agent_frames, text="Bot " + str(index + 1)).grid(row=0, column=0)
            self.agents[0].grid(row=1, column=0)
            self.add_agent_frame.grid(row=1, column=1)
        elif index == len(self.agents) - 2 and index < 8:
            ttk.Label(self.agent_frames, text="Bot " + str(index + 1)).grid(row=0, column=0)
            ttk.Label(self.agent_frames, text="Bot " + str(index + 2)).grid(row=0, column=1)
            self.agents[index].grid(row=1, column=0)
            self.agents[index + 1].grid(row=1, column=1)
            self.add_agent_frame.grid(row=1, column=2)
        else:
            ttk.Label(self.agent_frames, text="Bot " + str(index + 1)).grid(row=0, column=0)
            ttk.Label(self.agent_frames, text="Bot " + str(index + 2)).grid(row=0, column=1)
            ttk.Label(self.agent_frames, text="Bot " + str(index + 3)).grid(row=0, column=2)
            self.agents[index].grid(row=1, column=0)
            self.agents[index + 1].grid(row=1, column=1)
            self.agents[index + 2].grid(row=1, column=2)
        if index < 1 and self.left_button.winfo_ismapped():
            self.left_button.grid_forget()
        elif index != 0 and not self.left_button.winfo_ismapped():
            self.left_button.grid(row=2, column=0)
        if index > 6 and self.right_button.winfo_ismapped():
            self.right_button.grid_forget()
        elif not (index == 7 or index == len(self.agents) - 2) and not self.right_button.winfo_ismapped():
            self.right_button.grid(row=2, column=2)
        self.current_agent_frame = index


    class AgentFrame(tk.Frame):
        def __init__(self, parent, *args, **kwargs):
            tk.Frame.__init__(self, parent, *args, *kwargs)
            self.agent_path = tk.StringVar()
            self.config_options_path = {}

            # Agent path
            ttk.Label(self, text="Agent path: ", anchor="e").grid(row=0, column=0, padx=(10, 5), sticky="e")
            ttk.Entry(self, textvariable=self.agent_path, state="readonly").grid(row=0, column=1, sticky="ew")
            ttk.Button(self, text="Select file", command=self.change_bot_path).grid(row=0, column=2, padx=(5, 10),
                                                                                    sticky="ew")
            # Agent config
            ttk.Label(self, text="Bot Parameters: ", anchor="e").grid(row=1, column=0, padx=(10, 5), sticky="e")
            self.config_options = ttk.Combobox(self, state="readonly")
            self.config_options.bind("<<ComboboxSelected>>", self.change_config)
            self.config_options.grid(row=1, column=1, sticky="ew")
            self.config_button = ttk.Button(self, text="Add config", command=self.add_config_option, state="disabled")
            self.config_button.grid(row=1, column=2, padx=(5, 10), sticky="ew")

        def change_bot_path(self):
            trainer_file_path = askopenfilename(
                initialdir=os.path.dirname(os.path.realpath(__file__)),
                filetypes=[("Python File", "*.py")],
                title="Choose a file")
            if trainer_file_path:
                self.agent_path.set(trainer_file_path)
                self.config_button["state"] = "normal"
                self.config_options['values'] = ("custom",)
                self.config_options.set("custom")
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
                self.config_button["state"] = "normal"
                self.initialise_custom_config()
                self.change_config()

        def change_config(self):
            config_name = self.config_options.get()
            if config_name == "custom":
                if not self.custom_options.winfo_ismapped():
                    self.custom_options.grid(row=2, column=0, columnspan=3, sticky="ew")
            else:
                if self.custom_options.winfo_ismapped():
                    self.custom_options.grid_forget()

        def add_config_option(self):
            config_path = askopenfilename(
                initialdir=os.path.dirname(os.path.realpath(__file__)),
                filetypes=[("Config File", "*.cfg")],
                title="Choose a file.")
            if config_path:
                config_name = os.path.splitext(os.path.basename(os.path.realpath(config_path)))[0]
                self.config_options_path[config_name] = config_path
                self.config_options['values'] += (config_name,)
                self.config_options.set(config_name)

        def initialise_custom_config(self):
            self.custom_options = tk.Frame(self, borderwidth=2, relief=tk.SUNKEN)
            try:
                self.bot_config = self.agent_class[1].get_parameters_header()
            except AttributeError:
                error = "This class does not contain a config method, unable to create custom config"
                ttk.Label(self.custom_options, text=error).grid()
                return

            if not self.bot_config.values:
                ttk.Label(self.custom_options, text="No Bot Parameters for this agent").grid()

            ttk.Label(self.custom_options, text="Bot Parameters", anchor="center").grid(row=0, column=0, columnspan=2)
            for parameter_index, (parameter_name, parameter) in enumerate(self.bot_config.values.items()):
                ttk.Label(self.custom_options, text=parameter_name + ":", anchor='w').grid(
                    row=parameter_index + 1, column=0, sticky="ew", padx=(5, 5))
                big = 20000000
                if parameter.type == int:
                    parameter.value = tk.IntVar()
                    widget = tk.Spinbox(self.custom_options, textvariable=parameter.value, from_=0, to=big)
                elif parameter.type == float:
                    parameter.value = tk.DoubleVar()
                    widget = tk.Spinbox(self.custom_options, textvariable=parameter.value, from_=0, to=big, increment=.01)
                elif parameter.type == bool:
                    parameter.value = tk.BooleanVar()
                    widget = ttk.Combobox(self.custom_options, textvariable=parameter.value, values=(True, False),
                                          state="readonly")
                    widget.current(0) if not parameter.default else widget.current(1)
                elif parameter.type == str:
                    parameter.value = tk.StringVar()
                    widget = ttk.Entry(self.custom_options, textvariable=parameter.value)
                else:
                    widget = ttk.Label("Unknown type")

                widget.grid(row=parameter_index + 1, column=1, sticky="ew")

                if parameter.default is not None and parameter.type is not bool:
                    parameter.value.set(parameter.default)
            self.custom_options.grid_columnconfigure(1, weight=1)


if __name__ == '__main__':
    root = tk.Tk()
    runner = StartRunnerGUI(root)
    runner.pack(side="top", fill="both", expand=True)
    root.mainloop()
