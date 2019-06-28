# AI-Notes

Coursera Course: https://www.coursera.org/learn/introduction-tensorflow/home/welcome

Deeplearning.ai YouTube: https://www.youtube.com/channel/UCcIXc5mJsHVYTZR1maL5l9w/playlists

Google Colab: https://colab.research.google.com/notebooks/welcome.ipynb

Google Seedbank: https://research.google.com/seedbank/

TensorFlow: https://www.tensorflow.org/

TensorFlow Playground: http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.72727&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

## Neural Networks
intuition: make multiple guesses, which gradually approaches 100% accuracy (convergence)

**neural network:** set of functions that can learn patterns
- provide input and output data (labeled) => neural net provides relationship (fits inputs to outputs)
  - guesses diff relationship then loss function evaluates the guesses and passes info to optimizer for next guess
- simplest neural net has 1 neuron

**epoch:** 1 complete presentation of data set to neural net

**loss function:** measures accuracy of model during training
- need to minimize loss function for convergence

**optimizer:** how model is updated based on data and loss function

**metrics:** monitor training and testing (ex. accuracy)
ex.

```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

```


**Rule of Thumb:** first layer in your network should be the same shape as your data

**Rule of Thumb:** # of neurons in the last layer should match the number of classes you are classifying for
- i.e. last layer should match shape/form of desired output data

- deep learning => chaining simple layers
- layers extract representations from data fed into them
  - each layer req an activation function that instructs them
- most layers have params that are learned during training
- neural nets work best with normalized data (betw 0 and 1)
  - w/ python can just divide array

- neural net will not be exactly correct bec:
1. training it on finite dataset
2. neural nets use probability and adjust answers to fit

- neural net typically performs worse on unseen testing data vs training set

**Callback Function:** function called after each epoch to see if neural net has sufficient accuracy to end training
- usually call on_epoch_end since some algos have varying accuracy/loss during an epoch
- ex.

```
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
...
model = tf.keras.models.Sequential([
  ...
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
```

## Computer Vision
**Computer vision:** field of teaching computers to understand and label things present in an image

Intuition: feed large dataset of labelled images =>
computer figures out pattern

```
# ex. MNIST-fashion: dataset of 10 types of clothing (28x28 images)

# 3 layers in model
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)), # first layer matches input data (28x28 images) and converts 2D array to 1D array
  keras.layers.Dense(128, activation=tf.nn.relu), # hidden layer that figures out weights to assign
  keras.layers.Dense(10, activation=tf.nn.softmax) # last layer matches output (10 types of shoes)
  ])
```

- outputs an array of probabilities for each possible label
- adding more neurons => more calculations & higher accuracy
  - however, hits Law of Diminishing Returns very quickly  

https://www.tensorflow.org/tutorials/keras/basic_classification

## Convolutional Neural Network
Intuition: break image into diff components (ex. I see shoe laces and soles => shoe)
- some convolutions will change the image s.t. certain features in image are emphasized

- condense image into important features
  - don't look at each pixel
- pass filter over image to modify it
- look at pixel and neighbours
  - ex. 3x3 filter looks at current pixel and all its neighbours 1 pixel away
  - multiply all pixels in 3x3 grid by values in filter and then sum (1 convolution)

- ex. layer structure

1. Conv2D
2. MaxPooling2D
3. Conv2D
4. MaxPooling2D
5. Flatten
6. Dense  
7. Dense

- output of convolution is always smaller than input image
  - Conv2D only looks at full grids (does not operate on edges)

- convolution reduce amt of info => conv net can then extract meaningful info
- pooling compresses image  

**convolution:** filtering that highlights important features
- an operation where sum of products of elems from 2 2D functs, where 1 funct moves over every elem of other funct
- filter matrix must be symmetrical
  - otherwise it's a correlation, not a convolution
- filter must have odd size lengths (ex. 3x3, 5x5, etc) so that it has a centre


**pooling:** compressing image
- ex. go over image with a 2x2 grid and take the highest value at each point
- preserves features highlighted by convolutions

- need to reshape data for 1st convolution layer by combining multiple 3D arrays into a single 4D array
  - call `.reshape(NUM_Images, width, length, 1)` method

Note: If weights in filter don't sum to 0 or 1, should normalize them to get a sum of 1
- or else result will have diff brightness than original

sums:
1. if sum > 1   => brighter image
2. if sum == 1  => same brightness
3. if sum < 1   => darker image
4. if sum == 0  => very dark image

- for passing filters on image boundaries, can either crop, use values of zero, or wrap to other side (using modulo)
- usually force pixel values to be in [0,255]
  - can truncate or take absolute values

## ImageGenerator
- automatically generates labels for file containing sub directories
- subdirectory names are the labels



## Process
1. load data
2. Pre-process data
  - designate training & testing data
  - normalize
3. Design model
  - define layers, types of nodes, activation functions
4. Compile w/ optimizer and loss function
5. Test w/ testing data  

## Useful terms/objects in TensorFlow and Keras

**Sequential:** defines a sequence of layers in the neural network

**Flatten:** turns 2D array into 1D array

**Dense:** Adds a layer of connected neurons in Keras

**Conv2D:** 2D convolution layer (spatial convolution)

**MaxPooling2D:** max pooling layer for 2D inputs (ex. images)
- takes max value using a 2D grid

## Activation functions:

**Relu:** only passes values 0 or greater to the next layer in the network
- "If X>0 return X, else return 0"
- kinda like a diode

**Sigmoid:** logistic function that outputs a vector in range (0,1)

**Softmax:** takes an array of values (from prev layer) and replaces the **max elem** with 1 and the rest with 0
- ex [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05] => [0,0,0,0,1,0,0,0,0]

## Losses

**Cross-Entropy loss:** log loss

**Categorical Cross-Entropy loss:** Softmax activation +  Cross-Entropy loss

## Google Colab
- `shift enter`: runs code
- shareable python code on Google Drive


## Implementation Notes
- usually use part of dataset for training & rest for testing
- want to avoid introducing personal bias

- use `model.summary()` to examine layers

**overfitting:** network learns data from training set really well but becomes too specialized to that data and less effective for other data



## Future courses?
1. https://www.deeplearning.ai/ai-for-everyone/

## Good readings
1. Convolution Filters: https://lodev.org/cgtutor/filtering.html
2. Cross_entropy loss: https://gombru.github.io/2018/05/23/cross_entropy_loss/
