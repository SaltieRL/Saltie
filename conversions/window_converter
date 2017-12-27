from Tkinter import Tk, Label, Button


def create_popup(popup_text):
  app = Tk()
  app.title("SPIES")
  app.geometry("500x300+200+200")
  label1 = Label(app, text=popup_text, height=0, width=100)
  label1.pack()
  label2 = Button(app, text="Okay", width=20, command=app.destroy)
  label2.pack(side='bottom', padx=5, pady=5)
  app.mainloop()
