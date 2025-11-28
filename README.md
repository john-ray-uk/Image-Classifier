# Image Classifier

This is my first machine-learning program, an Image Classifier that uses PyTorch's CIFAR-10 dataset to allow the user to train the program's neural network (based on user inputs in its GUI), before testing it against
other images in the dataset. At the end of the program, it computes the neural network's accuracy against all 10,000 images of the dataset, as well as the accuracies of the program in each class.

It's taken me a week to code inbetween coursework and lectures, mainly involving me following [PyTorch's official beginner tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html), which is an excellent introduction to Deep Learning, and modifying the GUI that I used for my [Newtonian Potential Simulator program](https://github.com/john-ray-uk/Newtonian-Potential-Simulator) two weeks ago (as well as just borrowing some code from Stack Exchange).

It works well, and also features my first attempt at making a console GUI (updating the user as the program goes), and my first GUI that allows file inputs. I'm especially proud of the console tab of the GUI, which 
works by updating the console's text to be editable, appending a new entry to it, and then reupdating it to be read-only, each time it needs a new addition.

I mainly coded this for some internships I'm applying for at the moment, so if you're a recruiter for one of them - I hope this is convincing enough to recruit me! If I can teach myself Deep Learning from scratch and 
create an Image Classifier with a GUI in less than a week, in between coursework and lectures, imagine what I could do in 6 weeks under a full-time internship!
