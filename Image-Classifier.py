import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
import numpy as np
import torch.optim as optim
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
# Re-used code from my Newtonian Potential Simulator program to determine the base directory of the .ico file:
basedir = os.path.dirname(__file__)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def loadsets(batch,trainWorkers,testWorkers):
    global trainset
    global trainloader
    global testset
    global testloader
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,shuffle=True, num_workers=trainWorkers)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch,shuffle=False, num_workers=testWorkers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Default Neural Net conditions:
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# Code to un-normalise images from tensors to np.arrays, to show in program
def imshow(img):
    img = img / 2 + 0.5 # Un-normalisation
    npimg = img.numpy() # Convert to np.arrays
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # Plot the transposed result
    plt.show()

# Main UI code (some parts of the Neural Net code are in here, for functionality)
class app(tk.Tk):
    def __init__(self):
        super().__init__() # Keeps initial conditions set here in other methods like the UI.
        # Sets a title to the window and the window's size
        self.title("Image classifier, by John Ray")
        self.geometry("1920x768")
        # Define variables:
        self.batch = tk.IntVar(value=4)
        self.trainWorkers = tk.IntVar(value=0)
        self.testWorkers = tk.IntVar(value=0)
        self.momentum = tk.DoubleVar(value=0.9)
        self.learningrate = tk.DoubleVar(value=0.001)
        self.epoch = tk.IntVar(value=2)
        self.pathname = tk.StringVar(value="classifier_net")
        self.chosenpath = tk.StringVar(value='./classifier_net.pth')
        # Creates frames for the GUI (left-centres the buttons and such)
        blankframe = ttk.Frame(self, padding=0)
        blankframe.pack(side=tk.LEFT, fill=tk.Y,padx=50)
        self.frame = ttk.Frame(self, padding = 1)
        self.frame.pack(side=tk.LEFT, fill=tk.Y,expand=True, padx=5)
        loadsets(self.batch.get(),self.trainWorkers.get(),self.testWorkers.get())
        self.UI() # Loads the UI function.

    def UI(self):
        # Separating line (just to look nicer), from the ttk sub-package
        ttk.Separator(self.frame).pack(fill=tk.X, pady=10)
        # Title in the window
        Title = tk.Label(self.frame,text="Image Classifier")
        Title.config(font=("Helvetica", 12, "bold", "underline"))
        Title.pack()
        # Subtitle in the window
        SubTitle = tk.Label(self.frame,text="by John Ray")
        SubTitle.config(font=("Helvetica", 10))
        SubTitle.pack()
        # Another separator
        ttk.Separator(self.frame).pack(fill=tk.X, pady=5)
        # Ensures that only integers can be inserted
        def integerbatch(val):
            # Takes any float inputs and converts them to integer format
            val = int(float(val))
            return self.batch.set(val)
        # Buttons and menus (uses Tkinter's ttk sub-library):
        ttk.Label(self.frame,text="Number of images to reference off:").pack()
        ttk.Scale(self.frame, variable=self.batch,from_ = 1, to = 10, command=integerbatch).pack()
        ttk.Label(self.frame, textvariable = self.batch).pack()
        ttk.Separator(self.frame).pack(fill=tk.X, pady=5)
        ttk.Label(self.frame,text="Momentum of the model (0.9 by default):").pack()
        # Creates a scale for momentum, from 0.0 to 1.0
        def momentumround(val):
            val = f'{float(val):.02f}'
            return self.momentum.set(val)
        ttk.Scale(self.frame, variable=self.momentum, from_ = 0.0, to = 1.0,command=momentumround).pack()
        ttk.Label(self.frame, textvariable=self.momentum).pack()
        ttk.Separator()
        ttk.Label(self.frame,text="Learning rate of the model (0.001 by default):").pack()
        # Creates a scale for learning rate, from 0.0005 to 0.009
        def lrround(val):
            val = f'{float(val):.04f}'
            return self.learningrate.set(val)
        ttk.Scale(self.frame, variable=self.learningrate, from_ = 0.0005, to = 0.009,command=lrround).pack()
        ttk.Label(self.frame, textvariable=self.learningrate).pack()
        ttk.Separator()
        ttk.Label(self.frame,text="Number of iterations / epoch (2 by default):").pack()
        # Creates a scale for the number of iterations (epoch), from 1 to 10
        def iterround(val):
            val = int(float(val))
            return self.epoch.set(val)
        ttk.Scale(self.frame, variable=self.epoch, from_ = 1, to = 10,command=iterround).pack()
        ttk.Label(self.frame, textvariable=self.epoch).pack()
        ttk.Separator()
        # Subitle with an underline
        workers = ttk.Label(self.frame,text="Number of workers:")
        workers.config(font=("Helvetica",10,"underline"))
        workers.pack()
        ttk.Label(self.frame,text="Number of Train Workers (0 by default):").pack()
        # A command function for the chain size scale (ensures that only integer values can be inserted)
        def integertrainworkers(val):
            # Takes any float inputs and converts them to integer format
            val = int(float(val))
            # Sets this new value to be the value for the N Tkinter intVar
            return self.trainWorkers.set(val)
        ttk.Scale(self.frame, variable=self.trainWorkers,from_ = 0, to = 5, command=integertrainworkers).pack()
        ttk.Label(self.frame, textvariable=self.trainWorkers).pack()
        ttk.Label(self.frame,text="Number of Test Workers (0 by default):").pack()
        # A command function for the chain size scale (ensures that only integer values can be inserted)
        def integertestworkers(val):
            # Takes any float inputs and converts them to integer format
            val = int(float(val))
            # Sets this new value to be the value for the N Tkinter intVar
            return self.testWorkers.set(val)
        ttk.Scale(self.frame, variable=self.testWorkers,from_ = 0, to = 5, command=integertestworkers).pack()
        ttk.Label(self.frame, textvariable=self.testWorkers).pack()
        ttk.Separator(self.frame).pack(fill=tk.X, pady=5)
        ttk.Label(self.frame,text="Please choose a file name for the saved .pth model").pack()
        ttk.Entry(self.frame,textvariable=self.pathname).pack()
        ttk.Separator(self.frame).pack(fill=tk.X, pady=5)
        ttk.Label(self.frame,text="Click below to train the Image Classifier following the input parameters")
        ttk.Button(self.frame, text="Train classifier", command=lambda:(loadsets(self.batch.get(),self.trainWorkers.get(),self.testWorkers.get()),self.imageshow(),self.trainclassifier())).pack()
        ttk.Separator(self.frame).pack(fill=tk.X, pady=10)
        ttk.Label(self.frame,text='Please choose a .pth model file to test (defaults to recently created file):').pack()
        def fileDialog():
            filePath = filedialog.askopenfilename(title="Select a File", filetypes=[("PyTorch files", "*.pth")])
            if filePath:
                   fileLabel.config(text=f"Selected File: {filePath}")
            self.chosenpath.set(filePath)
            return
        fileLabel = tk.Label(self.frame, text="Selected File:")
        fileLabel.pack()
        tk.Button(self.frame, text="Open File", command=fileDialog).pack()
        ttk.Separator(self.frame).pack(fill=tk.X,pady=5)
        ttk.Label(self.frame,text="Click below to test the Image Classifier (please ensure that a .pth file exists beforehand)")
        ttk.Button(self.frame, text="Test classifier", command=lambda:(self.testclassifier())).pack()
        ttk.Separator(self.frame).pack(fill=tk.X, pady=5)
        # Creates a second right-centred frame for the graphs to be added to.
        plotframe = ttk.Frame(self, padding=8)
        plotframe.pack(side=tk.RIGHT, padx=25,pady=9)
        # Creates a figure
        ttk.Separator(plotframe).pack(fill=tk.X, pady=10)
        self.fig = plt.Figure()
        # Creates axis to append images
        self.ax = self.fig.add_subplot(111)
        # Enables the figures to be attached to a Tkinter "canvas" that enables Tkinter variables to be used.
        self.canvas = FigureCanvasTkAgg(self.fig, master=plotframe)
        # Fills in the canvas' information
        self.canvas.draw()
        # Unpacks (creates) the widget associated with Tkinter
        self.canvas.get_tk_widget().pack()
        # Does the same for the canvas widget.
        self.canvas._tkcanvas.pack()
        npimg = self.imageshow()
        # Adds a nice separating line to the bottom of the right-centred frame
        ttk.Separator(plotframe).pack(fill=tk.X, pady=10)
        # Creates a console frame, to show how the program is running
        c = ttk.Style()
        c.configure('Console.TFrame', background='#3e3f42', foreground="white", font=("Montserrat", 8))
        ctext = ttk.Style()
        ctext.configure('Ctext.TLabel', background='#3e3f42', foreground="white", font=("Montserrat", 10))
        ctitle = ttk.Style()
        ctitle.configure('Ctitle.TLabel', background='#3e3f42', foreground="white", font=("Montserrat", 10, "bold", "underline"))

        self.consoleframe = ttk.Frame(self, padding=10, style='Console.TFrame')
        self.consoleframe.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=50, pady=4)

        # Creates a title for this console frame
        ConsoleTitle = ttk.Label(self.consoleframe, text="Program Console:", style="Ctitle.TLabel")
        ConsoleTitle.pack()
        ttk.Separator(self.consoleframe).pack(fill=tk.X, pady=10)

        # Updating text for the console
        self.consoletext = tk.Text(self.consoleframe, height=12, wrap='word',
                                  bg='#3e3f42', fg='white', font=("Montserrat", 9), borderwidth=0)
        self.consoletext.pack(fill='both', expand=True)
        self.consoletext.insert('end', "UI established\n\n")
        self.consoletext.configure(state='disabled')  # Must be made read-only after each update

        return

    def imageshow(self):
         # Takes random images from the training set
        self.ax.clear()  
        dataiter = iter(trainloader)
        images, labels = next(dataiter)

        img_grid = torchvision.utils.make_grid(images)
        # Unnormalizes the images ((0.5,..),(0.5,..))
        img_grid = img_grid / 2 + 0.5
        # Converts to numpy and transposes
        npimg = img_grid.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        # Shows these images using standard matplotlib axes
        self.ax.imshow(npimg)
        # Creates a nice caption :)
        self.ax.set_title("Current training images: " + ', '.join(f'{classes[labels[j]]}' for j in range(self.batch.get())) + ".")
        self.canvas.draw()

    def trainclassifier(self):
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.learningrate.get(), momentum=self.momentum.get())
        self.consoletext.configure(state='normal')
        self.consoletext.insert('end', "Beginning iterations:\n\n")
        self.consoletext.see('end')
        self.consoletext.configure(state='disabled')
        self.update()
        for epoch in range((self.epoch.get())):  # loop over the dataset multiple times
            self.consoletext.configure(state='normal')
            self.consoletext.insert('end', f'Running iteration {epoch + 1}:\n\n')
            self.consoletext.see('end')
            self.consoletext.configure(state='disabled')
            self.update()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    self.consoletext.configure(state='normal')
                    self.consoletext.insert('end', f'For batches {i - 1999} to {i + 1} in iteration {epoch + 1}, average loss: {running_loss / 2000:.3f}\n\n')
                    self.consoletext.see('end')
                    self.consoletext.configure(state='disabled')
                    self.update()
                    running_loss = 0.0
        self.consoletext.configure(state='normal')
        self.consoletext.insert('end', "Training concluded!\n\n")
        self.consoletext.see('end')
        self.consoletext.configure(state='disabled')
        global PATH
        PATH = f'./{self.pathname.get()}.pth'
        torch.save(net.state_dict(), PATH)
        self.chosenpath = tk.StringVar(value=PATH)
        self.consoletext.configure(state='normal')
        self.consoletext.insert('end', f'Model saved to the file called "{self.pathname.get()}.pth" in the directory\n\n')
        self.consoletext.see('end')
        self.consoletext.configure(state='disabled')
    
    def testclassifier(self):
        # Randomly chooses a batch of images again, this time for testing
        self.ax.clear()
        self.consoletext.configure(state='normal')
        self.consoletext.insert('end', "Beginning training\n\n")
        self.consoletext.see('end')
        self.consoletext.configure(state='disabled')
        self.update()
        dataiter = iter(trainloader)
        images, labels = next(dataiter)

        img_grid = torchvision.utils.make_grid(images)
        # Unnormalize
        img_grid = img_grid / 2 + 0.5     # reverse the Normalize((0.5,..),(0.5,..))
        # Convert to numpy and transpose
        npimg = img_grid.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        self.ax.imshow(npimg)
        self.ax.set_title("Testing images show: " + ', '.join(f'{classes[labels[j]]}' for j in range(self.batch.get())) + ".")
        self.canvas.draw()
        self.consoletext.configure(state='normal')
        self.consoletext.insert('end', 'Testing images show: ' + ', '.join(f'{classes[labels[j]]}' for j in range(self.batch.get())) + "\n\n")
        self.consoletext.see('end')
        self.consoletext.configure(state='disabled')
        self.update()
        net = Net()
        net.load_state_dict(torch.load(self.chosenpath.get(), weights_only=True))
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        self.consoletext.configure(state='normal')
        self.consoletext.insert('end', 'Model thinks the images show: ' + ', '.join(f'{classes[predicted[j]]:5s}' for j in range(4)) + "\n\n")
        self.consoletext.see('end')
        self.consoletext.configure(state='disabled')
        self.update()
        correct = 0
        total = 0
        # Doesn't determine gradients, as it's testing again
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # Runs images through the network to determine outputs
                outputs = net(images)
                # Choose the prediction as the class with the highest "energy"
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # Print overall accuracy of the model across all test images
        self.consoletext.configure(state='normal')
        self.consoletext.insert('end', f'Accuracy of the network on all 10,000 test images: {100 * correct // total} %\n\n')
        self.consoletext.see('end')
        self.consoletext.configure(state='disabled')
        self.update()

        # Initialises dictionaries of correct and total predictions
        correctPred = {classname: 0 for classname in classes}
        totalPred = {classname: 0 for classname in classes}
        # Yet again, no gradients are needed, as it's testing
        with torch.no_grad():
            for data in testloader:
                # Same logic as earlier
                images, labels = data
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                # Collects the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correctPred[classes[label]] += 1
                    totalPred[classes[label]] += 1
        # Prints the accuracy of the model across each individual class
        for classname, correct_count in correctPred.items():
            accuracy = 100 * float(correct_count) / totalPred[classname]
            self.consoletext.configure(state='normal')
            self.consoletext.insert('end', f'Accuracy for the {classname} class is {accuracy:.1f} %\n\n')
            self.consoletext.see('end')
            self.consoletext.configure(state='disabled')
a = app()       
a.iconbitmap(os.path.join(basedir, "app.ico"))       

# Default code for closing Tkinter GUI
a.mainloop()