from tkinter.filedialog import askopenfilename
import tkinter as tk
from tkinter import ttk
import os
import importlib
import inspect
from bot_code.trainer.utils import custom_config
import runner as rocketleaguerunner


class StartRunnerGUI(tk.Frame):
    """Frame widget containing all necessary widgets to run bots."""
    def __init__(self, master, *args, **kwargs):
        """Construct a Frame with parent MASTER containing all widgets."""
        tk.Frame.__init__(self, master, *args, **kwargs)
        # Set parent settings
        self.parent = master
        self.parent.title("Runner GUI")
        self.parent.iconbitmap(default="images" + os.sep + "Saltie_logo.ico")

        self.blue_agents_book = ttk.Notebook(self)
        self.blue_agents = list()
        self.blue_agents_book.bind("<<NotebookTabChanged>>", lambda event: self.switch_tab(True))
        self.add_blue_agent_frame = tk.Frame(self.blue_agents_book)
        self.blue_agents_book.add(self.add_blue_agent_frame, text="Add Blue Agent")

        self.orange_agents_book = ttk.Notebook(self)
        self.orange_agents = list()
        self.orange_agents_book.bind("<<NotebookTabChanged>>", lambda event: self.switch_tab(False))
        self.add_orange_agent_frame = tk.Frame(self.orange_agents_book)
        self.orange_agents_book.add(self.add_orange_agent_frame, text="Add Orange Agent")

        self.add_agent(0, True)
        self.add_agent(0, False)

        self.blue_agents_book.grid(row=0, column=0)
        self.orange_agents_book.grid(row=0, column=1)

        self.start_button = ttk.Button(self, text="Start running!", command=self.start_running)
        self.start_button.grid(row=1, column=1, padx=(5, 10), pady=(5, 10), sticky="se")

    def get_team_info(self, blue):
        """Return the variables for the agents in that team."""
        return (self.blue_agents, self.blue_agents_book, self.add_blue_agent_frame) if blue \
            else (self.orange_agents, self.orange_agents_book, self.add_orange_agent_frame)

    def switch_tab(self, blue_team):
        """Handle tab switch to add an agent if switched to Add Agent tab"""
        agents_book = self.blue_agents_book if blue_team else self.orange_agents_book
        if all(s in agents_book.tab(agents_book.index("current"), option="text") for s in ["Add", "Agent"]):
            self.add_agent(agents_book.index("current"), blue_team)

    def add_agent(self, index, blue):
        """Add an agent to the according team."""
        agents, agents_book, add_agent_frame = self.get_team_info(blue)
        color = "Blue" if blue else "Orange"
        agents.append(self.AgentFrame(agents_book))
        agents_book.insert(index, agents[index], text=color + " Bot " + str(index + 1))
        if len(agents) > 4:
            agents_book.hide(add_agent_frame)
        agents_book.select(index)

    def update_tabs(self, blue):
        agents, agents_book, add_agent_frame = self.get_team_info(blue)
        color = "Blue" if blue else "Orange"
        for i, widget in reversed(list(enumerate(agents_book.tabs()))):
            label = agents_book.tab(widget, option="text")
            if not label.endswith(str(i + 1)) and not all(s in label for s in ["Add", "Agent"]):
                agents_book.tab(widget, text=color + " Bot " + str(i + 1))

    def remove_agent(self, agent):
        """Remove agent AGENT from the team list containing AGENT."""
        if agent in self.blue_agents:
            agents, agents_book, add_agent_frame = self.get_team_info(True)
            blue = True
        elif agent in self.orange_agents:
            agents, agents_book, add_agent_frame = self.get_team_info(False)
            blue = False
        else:
            return
        agents_book.hide(add_agent_frame)
        agent.destroy()
        agents.remove(agent)
        if len(agents) == 0:
            self.add_agent(0, blue)
        agents_book.add(add_agent_frame)
        self.update_tabs(blue)

    def start_running(self):
        """Start runner.py with the appropriate configuration after running the injector"""
        os.system(os.path.dirname(os.path.realpath(__file__)) + os.sep + "RLBot_Injector.exe hidden")
        num_bots = len(self.blue_agents) + len(self.orange_agents)

        rlbotcfg = custom_config.ConfigObject()
        cfg_header = rlbotcfg.add_header_name('RLBot Configuration')
        cfg_header.add_value("num_participants", int, var=num_bots)

        participiant_header = rlbotcfg.add_header_name('Participant Configuration')
        i = 0
        agent_locations = []
        for t, team in enumerate([self.blue_agents, self.orange_agents]):
            for agent in self.blue_agents:
                participiant_header.add_value("participant_config_" + str(i), str, var=agent.looks_path)
                participiant_header.add_value("participant_team_" + str(i), str, var=t)
                participiant_header.add_value("participant_is_bot_" + str(i), bool, var=agent.is_bot)
                participiant_header.add_value("participant_is_rlbot_controlled_" + str(i), bool,
                                              var=agent.rlbot_controlled)
                level = agent.bot_level.get()
                skill = 0.5 if level == "Pro" else 1.0 if level == "All-Star" else 0
                participiant_header.add_value("participant_bot_skill_" + str(i), float, var=skill)
                agent_locations.append(agent.agent_path.get())
                i += 1

        self.forget()
        self.parent.destroy()
        rocketleaguerunner.main(framework_config=rlbotcfg, bot_location=agent_locations)

    class AgentFrame(tk.Frame):
        """Frame widget all necessary widgets for an Agent."""
        def __init__(self, parent, *args, **kwargs):
            """Construct a Frame with parent MASTER containing all widgets."""
            tk.Frame.__init__(self, parent, *args, *kwargs)
            self.config(relief="flat", borderwidth=4)
            self.agent_path = tk.StringVar()
            self.config_options_path = {}
            self.looks_path = tk.StringVar()
            self.bot_config = None
            self.agent_class = None

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
            self.is_bot_widgets[1].bind("<<ComboboxSelected>>", lambda e: self.change_is_bot())
            self.is_bot_widgets[1].current(0)

            self.rlbot_controlled = tk.BooleanVar()  # row 2
            self.rlbot_controlled_widgets = list()
            self.rlbot_controlled_widgets.append(ttk.Label(self, text="RLBot controlled: ", anchor="e"))
            self.rlbot_controlled_widgets.append(
                ttk.Combobox(self, textvariable=self.rlbot_controlled, values=(True, False), state="readonly"))
            self.rlbot_controlled_widgets[1].bind("<<ComboboxSelected>>", lambda e: self.change_rlbot_controlled())
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
            self.agent_config_widgets[1].bind("<<ComboboxSelected>>", lambda e: self.change_config())
            self.agent_config_widgets.append(
                ttk.Button(self, text="Add config", command=self.add_config_option, state="disabled"))

            self.custom_agent_options = tk.Frame(self, borderwidth=2, relief=tk.SUNKEN)  # row 7

            ttk.Button(self, text="Remove", command=lambda: parent.master.remove_agent(self)).grid(row=8, column=2)

            self.grid_items(0, 0, self.looks_widgets, self.is_bot_widgets)
            self.change_is_bot()
            self.change_rlbot_controlled()

        @staticmethod
        def grid_items(start_row=0, start_index=0, *widgets):
            """Grid all items found in WIDGETS starting in row START_ROW from index START_INDEX"""
            for row, widget_list in enumerate(widgets):
                row += start_row
                for column, widget in enumerate(widget_list):
                    column += start_index
                    widget.grid(row=row, column=column, sticky="nsew")

        def change_is_bot(self, hide=False):
            """Hide or show the widgets which have to do with the is_bot value."""
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

        def change_rlbot_controlled(self, hide=False):
            """Hide or show the widgets which have to do with the rlbot_controlled value."""
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
            """Popup, ask for a new agent path and apply that path."""
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
            """Popup, ask for the loadout config and apply that path."""
            config_path = askopenfilename(
                initialdir=os.path.dirname(os.path.realpath(__file__)),
                filetypes=[("Config File", "*.cfg")],
                title="Choose a file")
            if config_path:
                self.looks_path.set(config_path)

        def change_config(self):
            """Handle ComboboxSelected Event and show/hide the Custom Frame."""
            config_name = self.agent_config_widgets[1].get()
            if config_name == "custom":
                if not self.custom_agent_options.winfo_ismapped():
                    self.custom_agent_options.grid(row=6, column=0, columnspan=3, sticky="nsew")
            else:
                if self.custom_agent_options.winfo_ismapped():
                    self.custom_agent_options.grid_forget()

        def add_config_option(self):
            """Popup, ask for an extra config option and add it to the combobox"""
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
            """Create the Custom Config Frame containing all widgets for editing the parameters."""
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
