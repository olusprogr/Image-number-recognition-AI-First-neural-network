import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class LossGUI:
    def __init__(self, title="Loss Visualization"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("600x400")
        self.firstloss = 0
        self.firstloss_set = False

        self.loss_values = []
        self.cross_entropy_values = []

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_title("Training Loss per Epoch")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)



    def update_plot(self, loss, cross_entropy=False):
        if loss == 0:
            return
        
        if not self.firstloss_set:
            self.firstloss = loss
            self.firstloss_set = True

        self.loss_values.append(loss)
        if cross_entropy:
            self.cross_entropy_values.append(cross_entropy)
        self.ax.clear()

        self.ax.axhspan(1.0, self.firstloss, facecolor='red', alpha=0.3)
        self.ax.axhspan(0.5, 1.0, facecolor='yellow', alpha=0.3)
        self.ax.axhspan(0.2, 0.5, facecolor='lightgreen', alpha=0.3)
        self.ax.axhspan(0.0, 0.2, facecolor='darkgreen', alpha=0.5)

        self.ax.plot(self.loss_values, marker='o', color='blue', label='Loss')
        self.ax.plot(len(self.loss_values)-1, loss, marker='o', color='red', markersize=10, label='Aktueller Loss')

        if cross_entropy is not None:
            self.ax.plot(self.cross_entropy_values, marker='x', color='orange', label='Accuracy so far')
            self.ax.set_title("Training Loss and Accuracy per Epoch")
        else:
            self.ax.set_title("Training Loss per Epoch")

        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True)
        self.ax.legend()

        self.ax.axhline(y=1, color='black', linestyle='--', linewidth=1)

        self.canvas.draw()
        self.root.update_idletasks()
        self.root.update()


    def start(self):
        self.root.mainloop()

    def close(self):
        self.root.destroy()
