# ImageNet Classification with Deep Convolutional Neural Networks Notes


# Abstract & Intro
- 60 M params
- 650 000 neurons
- 5 CONV Layers
- Max pooling
- 3 FC Layers
- Final 1000-way softmax
- used dropout regularization method to avoid overfitting
- 5-6 days trianing on two GTX 580 3GB GPUs


# Dataset
- ImageNet has > 15 M labeled high res images w/ 22 000 categories
  - labelled w/ Amazon Mech Turk
- annual ImageNet Large-Scale Visual Rec Challenge (ILSVRC) uses subset of ImageNet w/ 1000 images in 1000 categories
- total 1.2 M training images, 50K validation images, 150K testing images
- report 2 error rates on ImageNet: top-1 and top-5
  - top-5 error rate is fraction of test images for which correct label not among five labels considered most probable by model
- ImageNet has variable-resolution images => they down-sampled images to 256x256 bec sys req constant input dimensionality
  - for rectangular image, rescaled image s.t. shorter side was L=256, then cropped out central 256x256 patch from resulting image
- no other pre-processing except for subtracting mean activity over training set from each pixel
- trained network on (centered) raw RGB values of pixels


## Architecture
- 8 learned layers (5 CONV, 3 FC)

### ReLU Nonlinearity
- standard model of neuron output f of input x is f(x) = tanh(x) or f(x) = (1+exp(-x))^-1
  - these saturating nonlinearities are slower than non-saturating nonlinearity f(x) = max(0,x) in training time

**Rectified Linear Units (ReLU):** neurons w/ nonlinearity f(x) = max(0,x)
 - deep conv nets w/ ReLUs train faster than equivalent tanh units
 - nonlinearity f(x)=|tanhx| works well w/ contrast normalization w/ local avg pooling
    - good at preventing overfitting, but slower

### Training on Multiple GPUs    
- current GPUs good at cross-GPU parallelization
  - can read/write to another's memory directly w/out going thru host machine memory
- their parallelization scheme puts half of kernels (neurons) on each GPU
- their GPUs only communicate on certain layers
    - ex. kernels of layer 3 take input form all kernel maps in layer 2, but kernels in layer 4 take input only form kernel maps in layer 3 which are on same GPU
    - reduces error rates and training time
- similar to columnar CNN, except their columns not indep


### Local Response Normalization
- ReLU's good bec do not req input normalization to prevent oversaturation
- if at least some training ex produce positive input to a ReLU, learning will happen in that neuron
- adding local normalization schemes aids generalization
  - used "brightness normalization" to reduce error

### Overlapping Pooling
- pooling layers in CNNs summarize outputs of groups of neurons in same kernel map
- POOL layer like grid of pooling units spaced s units apart, each summarizing a zxz neighbourhood  centered at location of unit
- if:
1. s=z: traditional local pooling
2. s<z: overlapping pooling
- they used overlapping pooling w/ s=2, z=3
  - slightly reduced error and harder to overfit


### Overall Architecture
- INPUT -> 5 CONV -> 3 FC -> 1000-way softmax
- maximizes multinomial logistic regression objective
  - equiv to maximizing average across training cases of log-probability of correct label under prediction distribution
- kernels of 2nd, 4th, 5th CONV layers only connected to kernel maps in prev layer on same GPU
  - kernels in 3rd CONV fully connected to all kernel maps in 2nd layer
- response-normalization follow 1st and 2nd CONV layers
- Max pooling follow both response-normalization and 5th CONV
- ReLU non-linearity applied to output of all CONV and FC
- CONV Layers
1. 1st CONV filters 224x224x3 image w/ 96 kernels (11x11x3) w/ stride 4 pixels
2. 2nd CONV takes response-normalized and pooled output of 1st CONV and filters it w/ 256 kernels (5x5x48)
- 3rd, 4th, 5th layers connected w/ no pooling or normalization layers
3. 3rd CONV has 384 kernels (3x3x256) connected to response-normalized & pooled output of 2nd CONV
4. 4th CONV has 384 kernels (3x3x192)
5. 5th CONV has 384 kernels (3x3x192)
- FC layers have 4096 neurons


## Reducing Overfitting
- 60M params, 1000 classes (10 bits of constraint on image-label mapping)
- cannot learn this many params w/out lots of overfitting

### Data Augmentation
- easiest to avoid overfitting by artifically enlarging dataset w/ label-preserving transformations
- they used 2 forms of data augmentation
  - transformed images generated on CPU while GPU training on prev set of training images
    - no need to store transformed images
    - effectively computationally free

- Data augmentation:
1. generating image translations & reflections
- extract random 224x224 patches (and horizontal reflections) from 256x256 images and train on extracted patches
- increases size of training set by factor of 2048, but resulting training examples are very interdep   
- w/out scheme have high overfitting
- at test time, network makes prediction by selecting 10 patches (4 corners, 1 centre, + 5 reflections) and avging predictions on patches using softmax
2. altering RGB channel intensities
- perform PCA (principal component analysis) on set of RGB pixel values
- add multiples of principal components w/ magnitudes proportional to corresponding eigenvalues times a random var from Gaussian w/ mean 0 and standard dev 0.1
- scheme reduces error bec object identity is invariant to changes in intensity and color of illumination


## Dropout
- can reduce test errors by combining predictions of diff models, but very time expensive
**dropout:** consists of zeroing output of each hidden neuron w/ probability 0.5
- very efficient model combo (factor of 2 during training)
- dropped out neurons do not contribute to forward pass or backpropagation
- every new input, neural net samples a diff architecture but all architecture shares weights
- reduces complex co-adaptations of neurons since a neuron cannot rely on presence of particular neurons
  - forced to learn more robust features that are useful in conjunction with other random subsets of neurons
- at test time, use all neurons, but multiply outputs by 0.5 (approx of taking geom mean of predictive distros produced by eponentially-many dropout networks)
- use dropout in first 2 FC
- dropout doubles \# of interations req to converge


## Details of Learning
- trained models w/ stochastic gradient descent w/ batch size of 128, momentum of 0.9, and weight decay of 0.0005
- small weight decay reduces training error
- init weights in each layer from zero-mean Gaussian distro w/ stdev 0.01
- init neuron biases  in 2nd, 4th, 5th CONV layers and FC hidden layers w/ constant 1  
  - accelerates early learning stages by providing ReLUs w/ positive inputs
- init neuron biases in other layers w/ constant 0
- equal learning rate for all layers (init 0.01)
  - divided learning rate by 10 when validation error rate stopped improving w/ current learning rate
  - reduced learning rate 3 times before termination
- trained network approx 90 cycles 
