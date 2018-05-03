# Project description: 
In this project, there are 4000 labeled images as cat and dog. The set is divided into training, validation and test sets of 1000, 500 and 500 examples of each class, respectively.
                  
A pretrained convolutional neural network known as VGG16 is used to extract features of each image in the small set of cat and dogs.

After extracting features of all images in the dataset, they are fed into a fully connected layer of 256 hidden units. Dropout with probability of 0.5 is used for the fully connected layer and then the result is fed to output layer with sigmoid activation. Since original VGG16 was introduced for ImageNet problem with 1000 classes, the last layer is changed. 

# Modification: 
To increase the accuracy, data augmentation can be used. To this end, the output of the VGG16 model is first falttened as a vector and then fed into a fully connected layer of 256 units and finally it is is connected to output layer. Freezing the convolutional base and then training the entire end-to-end deep neural network is done using GPU.

# Fine-tuning:
To further improve the accuracy, the trained model in the previous step is used again and then the last layer in the convolutional base is unfreezed and these layers together with fully connected layers are trained again.
