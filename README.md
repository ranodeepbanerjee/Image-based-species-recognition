# Cat vs Dog Image Classifier
This is a simple image classification project that uses a Convolutional Neural Network (CNN) to classify whether an input image contains a cat or a dog.<br>
[Kaggle Notebook Link](https://www.kaggle.com/ranodeepbanerjee/dogs-cats-project)

## Requirements
TensorFlow 2.0 or higher<br>
Keras<br>
NumPy<br>
matplotlib<br>
OpenCV<br>

## Dataset
The dataset used for this project is the [Cat and Dog dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog) which is available on Kaggle. The dataset contains 8000 images of cats and dogs (4000 images of each class) for training, and 2000 images of cats and dogs (1000 images of each class) for testing.

## Model Architecture
### The model architecture is a simple CNN with the following layers:
1.Convolutional layer with 32 filters and a kernel size of 3x3<br>
2.ReLU activation layer<br>
3.Max pooling layer with a pool size of 2x2 and stride of 2<br>
4.Flatten layer<br>
5.Dense layer with 128 neurons and ReLU activation function<br>
6.Output layer with 1 neuron and sigmoid activation function

## Training
The model is trained using the binary_crossentropy loss function and the adam optimizer for 25 epochs. The training data is augmented using various transformations such as shear range, zoom range, and horizontal flip.

## Testing
The trained model is tested on a separate set of 2000 images (1000 images of each class). The test accuracy of the model is evaluated.

## Author
This project was created by [Ranodeep Banerjee](https://github.com/ranodeepbanerjee). Feel free to contact me if you have any questions or suggestions.

