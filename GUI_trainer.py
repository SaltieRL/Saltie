from tkinter.filedialog import askopenfilename
import tkinter as tk
from tkinter import ttk
import os

import importlib
import inspect

class TrainerGUI(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.trainer_path = tk.StringVar()

        tk.Label(self, text="Trainer path: ").grid(row=0, column=0)
        tk.Entry(self, textvariable=self.trainer_path).grid(row=0, column=1, sticky="ew")
        tk.Button(self, text="Select file", command=self.change_trainer_path).grid(row=0, column=2)

        tk.Label(self, text="Config: ").grid(row=1, column=0)
        self.config_options = ttk.Combobox(self, state="readonly")
        self.config_options.bind("<<ComboboxSelected>>", self.change_config)
        self.config_options.grid(row=1, column=1, sticky="ew")
        self.config_button = tk.Button(self, text="Add config", command=self.add_config_option, state="disabled")
        self.config_button.grid(row=1, column=2)

    def initialise_custom_config(self):
        self.custom_options = tk.Frame(self)
        layout = self.trainer_class[1]().config_layout
        self.trainer_config = {}
        for header_index, header in enumerate(layout.headers):
            if not header.values:
                continue
            header_frame = tk.Frame(self.custom_options)
            header_frame.grid(row=0, column=header_index)
            tk.Label(header_frame, text=header.name).grid(row=0, column=0, columnspan=2)
            self.trainer_config[header.name] = {}
            for i, parameter in enumerate(header.values):
                tk.Label(header_frame, text=parameter.name).grid(row=i + 1, column=0)
                if parameter.type == int:
                    self.trainer_config[header.name][parameter.name] = tk.IntVar()
                    tk.Spinbox(header_frame, textvariable=self.trainer_config[header.name][parameter.name]).grid(row=i + 1, column=1)
                elif parameter.type == float:
                    self.trainer_config[header.name][parameter.name] = tk.DoubleVar()
                    tk.Spinbox(header_frame, textvariable=self.trainer_config[header.name][parameter.name]).grid(row=i + 1, column=1)
                elif parameter.type == bool:
                    print("Gotta add boolean thing")
                    self.trainer_config[header.name][parameter.name] = tk.BooleanVar()
                elif parameter.type == str:
                    self.trainer_config[header.name][parameter.name] = tk.StringVar()
                    tk.Entry(header_frame, textvariable=self.trainer_config[header.name][parameter.name]).grid(row=i + 1, column=1)
                else:
                    print("UNKNOWN TYPE")
                if parameter.default is not None:
                    self.trainer_config[header.name][parameter.name].set(parameter.default)



    def change_trainer_path(self):
        trainer_file_path = askopenfilename(
            initialdir=os.path.dirname(os.path.realpath(__file__)) + os.sep + "trainer",
            filetypes=[("Python File", "*.py")],
            title="Choose a file.")
        if trainer_file_path:
            self.trainer_path.set(trainer_file_path)
            self.config_button["state"] = "normal"
            trainer_name = os.path.splitext(os.path.basename(os.path.realpath(trainer_file_path)))[0]
            config_path = os.path.dirname(
                os.path.realpath(__file__)) + os.sep + "trainer" + os.sep + "configs" + os.sep + trainer_name + ".cfg"
            if os.path.isfile(config_path):
                self.default_config_path = config_path
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
                tk.Label(popup, text="Select the class and press continue").grid(row=0, column=0, columnspan=2, padx=(10, 10), pady=(10, 5))
                for i in range(len(trainer_classes)):
                    tk.Radiobutton(popup, text=trainer_classes[i][0], value=i, anchor="w", variable=selected).grid(
                        row=i + 1, column=0, sticky="nsew", padx=(10, 0))
                selected.set(0)

                def chosen_class():
                    self.trainer_class = trainer_classes[selected.get()]
                    popup.destroy()

                tk.Button(popup, text="Continue", anchor="se", command=chosen_class).grid(row=len(trainer_classes), column=1, padx=(0, 10), pady=(0, 10))
                self.wait_window(popup)
            else:
                self.trainer_class = trainer_classes[0]
            self.initialise_custom_config()
            self.change_config()

    def add_config_option(self):
        config_path = askopenfilename(
            initialdir=os.path.dirname(os.path.realpath(__file__)) + os.sep + "trainer" + os.sep + "configs",
            filetypes=[("Config File", "*.cfg")],
            title="Choose a file.")
        if config_path:
            config_name = os.path.splitext(os.path.basename(os.path.realpath(config_path)))[0]
            self.config_options['values'] += (config_name,)
            self.config_options.set(config_name)

    def change_config(self, event=None):
        config_name = self.config_options.get()
        if config_name == "custom":
            if not self.custom_options.winfo_ismapped():
                self.custom_options.grid(row=2, column=0, columnspan=3)
                print("Add custom stuff")
        else:
            if self.custom_options.winfo_ismapped():
                self.custom_options.grid_forget()
                print("Remove custom stuff")



if __name__ == '__main__':
    root = tk.Tk()
    TrainerGUI(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
