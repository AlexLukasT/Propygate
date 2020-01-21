# Propygate

### A Deep Learning framework based on pure numpy


I implemented a neural network from scratch using only numpy (and some keras but only for loading the MNIST dataset).   
The idea behind this project is not to be competitive with existing deep learning frameworks but to get a personal insight in the "magic" of deep neural networks.   
A lot of help and inspirition was found on http://neuralnetworksanddeeplearning.com/.
With just basic gradient descent and no finetuning of any parameters one can actually achieve a reasonable performance on the MNIST dataset:

![Imgur](https://i.imgur.com/w3YfRoD.png)

Some examples:

![Imgur](https://i.imgur.com/WDiyYL7.png)
![Imgur](https://i.imgur.com/8SRat9g.png)
![Imgur](https://i.imgur.com/ysSx45Q.png)

Future plans:  
* Finish modularization (a class for each layer, optimizer, activation, ...)
* Add more initializers and optimizers
* Add Convolutions (including backpropagation)
