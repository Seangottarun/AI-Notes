# AI-Notes

Coursera Course: https://www.coursera.org/learn/introduction-tensorflow/home/welcome

Google Colab: https://colab.research.google.com/notebooks/welcome.ipynb

Google Seedbank: https://research.google.com/seedbank/

TensorFlow: https://www.tensorflow.org/

TensorFlow Playground: http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.72727&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

## Neural Networks
logic: make multiple guesses, which gradually approaches 100% accuracy (convergence)

**neural network:** set of functions that can learn patterns
- provide input and output data (labeled) => neural net provides relationship (fits inputs to outputs)
  - guesses diff relationship then loss function evaluates the guesses and passes info to optimizer for next guess
- simplest neural net has 1 neuron

**Dense**: defines a layer of connected neurons in Keras

- neural net will not be exactly correct bec:
1. training it on finite dataset
2. neural nets use probability and adjust answers to fit

- ex. using mean squared error for loss function and SGD (stochastic gradient descent) for optimizer

```
# ex. relationship is y = 0.5x +0.5

import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

xs = np.array([1,2,3,4,5,6], dtype=float)
ys =np.array([1,1.5,2,2.5,3,3.5], dtype=float)

model.fit(xs, ys, epochs = 500)
print(model.predict([7.0])) # expect: 4

# for 500 epochs, output: 4.0126038, loss: 7.6350e-05
# for 1000 epochs, output: 3.9998424, loss: 1.1911e-08
```

## Google Colab
- `shift enter`: runs code
- shareable python code on Google Drive

## Examples
1. https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=oxNzL4lS2Gui

## Future courses?
1. https://www.deeplearning.ai/ai-for-everyone/
