### `./nn/` -
To get used to training a simple neural network, a simple PoC to -
1. Create a simple 2-layered-neural-network
2. Train the network in 1000 epochs.
3. In each epoch -
    
    a. Feed forward: training data is passed through input & hidden layers. Predictions are collected at output layer 
    
    b. Compute loss, perform back propagation, update weights
4. Test data is passed through trained model (feed forward) & predictions are collected at output layer.
5. Accuracy is computed by comparing predicted data with ground truth.
6. Loss function used: cross entropy
7. Optimizer used to find minima of loss function: SGD
8. ReLU is applied between input & hidden layer to induce non-linearity in classification boundary. It also helps us in mitigating vanishing gradient issue.
### `./cnn/`
Getting started with training CNNs to understand -
1. How conv2d layers are used to perform convolution using kernels
2. How Maxpool layers help to reduce feature map size, ignoring noise by capturing only maximum value in a pooling region
3. Typical CNN architecture where initial conv layers capture features from an image & subsequent FC layers find patterns in these features
4. How shape (dimensions) of feature map changes after convolution & max pooling.
