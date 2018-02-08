from tkinter.filedialog import askopenfilename
import tkinter as tk
from tkinter import ttk
import os

import importlib
import inspect

from bot_code.trainer.utils import trainer_runner


class StartTrainerGUI(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.title("Trainer GUI")
        self.parent.iconbitmap(default="images" + os.sep + "Saltie_logo.ico")

        self.trainer_path = tk.StringVar()
        self.config_options_path = {}

        ttk.Label(self, text="Trainer path: ", anchor="e").grid(row=0, column=0, padx=(0, 5), sticky="e")
        ttk.Entry(self, textvariable=self.trainer_path, state="readonly").grid(row=0, column=1, sticky="ew")
        ttk.Button(self, text="Select file", command=self.change_trainer_path).grid(row=0, column=2, padx=(5, 0),
                                                                                    sticky="w")

        ttk.Label(self, text="Config: ", anchor="e").grid(row=1, column=0, padx=(0, 5), sticky="e")
        self.config_options = ttk.Combobox(self, state="readonly")
        self.config_options.bind("<<ComboboxSelected>>", self.change_config)
        self.config_options.grid(row=1, column=1, sticky="ew")
        self.config_button = ttk.Button(self, text="Add config", command=self.add_config_option, state="disabled")
        self.config_button.grid(row=1, column=2, padx=(5, 0), sticky="w")

        self.start_button = ttk.Button(self, text="Start training!", command=self.start_training, state="disabled")
        self.start_button.grid(row=3, column=2, sticky="se")

    def initialise_custom_config(self):
        self.custom_options = tk.Frame(self, borderwidth=2, relief=tk.SUNKEN)
        try:
            self.trainer_config = self.trainer_class[1](load_config=False).config_layout
        except AttributeError as e:
            error = "This class does not contain a config layout, unable to create custom config"
            ttk.Label(self.custom_options, text=error).grid()
            return
        header_index = 0
        for header_name, header in self.trainer_config.headers.items():
            if not header.values:
                continue
            header_frame = tk.Frame(self.custom_options)
            header_frame.grid(row=0, column=header_index, sticky="n", padx=(10, 10), pady=(10, 10))
            ttk.Label(header_frame, text=header_name, anchor="center").grid(row=0, column=0, columnspan=2)
            parameter_index = 0
            for parameter_name, parameter in header.values.items():
                ttk.Label(header_frame, text=parameter_name + ":", anchor='w').grid(row=parameter_index + 1, column=0,
                                                                                    sticky="ew")
                big = 20000000
                if parameter.type == int:
                    parameter.value = tk.IntVar()
                    tk.Spinbox(header_frame, textvariable=parameter.value, from_=0, to=big).grid(
                        row=parameter_index + 1, column=1, sticky="ew")

                elif parameter.type == float:
                    parameter.value = tk.DoubleVar()
                    tk.Spinbox(header_frame, textvariable=parameter.value, from_=0, to=big, increment=.01).grid(
                        row=parameter_index + 1, column=1, sticky="ew")

                elif parameter.type == bool:
                    parameter.value = tk.BooleanVar()
                    box = ttk.Combobox(header_frame, textvariable=parameter.value, values=(True, False),
                                       state="readonly")
                    if not parameter.default:
                        box.current(1)
                    else:
                        box.current(0)
                    box.grid(row=parameter_index + 1, column=1, sticky="ew")

                elif parameter.type == str:
                    parameter.value = tk.StringVar()
                    ttk.Entry(header_frame, textvariable=parameter.value).grid(row=parameter_index + 1, column=1,
                                                                               sticky="ew")

                else:
                    print("Unknown type for", parameter_name, "in", self.trainer_class[0])

                if parameter.default is not None and parameter.type is not bool:
                    parameter.value.set(parameter.default)
                parameter_index += 1

            header_index += 1

    def change_trainer_path(self):
        trainer_file_path = askopenfilename(
            initialdir="bot_code" + os.sep + "trainer",
            filetypes=[("Python File", "*.py")],
            title="Choose a file.")
        if trainer_file_path:
            self.trainer_path.set(trainer_file_path)
            self.config_button["state"] = "normal"
            trainer_name = os.path.splitext(os.path.basename(os.path.realpath(trainer_file_path)))[0]
            config_path = "bot_code" + os.sep + "trainer" + os.sep + "configs" + os.sep + trainer_name + ".cfg"
            if os.path.isfile(config_path):
                self.default_config_path = config_path
                self.config_options_path[trainer_name] = config_path
                self.config_options['values'] = (trainer_name, "custom")
                self.config_options.set(trainer_name)
            else:
                self.config_options['values'] = ("custom",)
                self.config_options.set("custom")
            module = self.trainer_path.get().replace(os.path.dirname(os.path.realpath(__file__)).replace(os.sep, "/"),
                                                     "", 1).replace("/", ".")[1:-3]
            trainer_package = importlib.import_module(module)
            trainer_classes = [m for m in inspect.getmembers(trainer_package, inspect.isclass) if
                               m[1].__module__ == module]
            if len(trainer_classes) > 1:
                popup = tk.Toplevel()
                popup.title("Choose trainer class")
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
                    self.trainer_class = trainer_classes[selected.get()]
                    popup.destroy()

                ttk.Button(popup, text="Continue", command=chosen_class).grid(row=len(trainer_classes), column=1,
                                                                              padx=(0, 10), pady=(0, 10))
                self.wait_window(popup)
            else:
                self.trainer_class = trainer_classes[0]
            self.start_button["state"] = "normal"
            self.initialise_custom_config()
            self.change_config()

    def add_config_option(self):
        config_path = askopenfilename(
            initialdir="bot_code" + os.sep + "trainer" + os.sep + "configs",
            filetypes=[("Config File", "*.cfg")],
            title="Choose a file.")
        if config_path:
            config_name = os.path.splitext(os.path.basename(os.path.realpath(config_path)))[0]
            self.config_options_path[config_name] = config_path
            self.config_options['values'] += (config_name,)
            self.config_options.set(config_name)

    def change_config(self, event=None):
        config_name = self.config_options.get()
        if config_name == "custom":
            if not self.custom_options.winfo_ismapped():
                self.custom_options.grid(row=2, column=0, columnspan=3)
        else:
            if self.custom_options.winfo_ismapped():
                self.custom_options.grid_forget()

    def start_training(self):
        if self.config_options.get() == "custom":
            trainer_class = self.trainer_class[1](config=self.trainer_config)
        else:
            trainer_class = self.trainer_class[1](config_path=self.config_options_path[self.config_options.get()])
        self.forget()
        self.parent.destroy()
        trainer_runner.run_trainer(trainer_class)


if __name__ == '__main__':
    root = tk.Tk()
    StartTrainerGUI(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
