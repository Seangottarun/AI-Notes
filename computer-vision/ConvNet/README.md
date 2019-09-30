# Convolutional Neural Nets

**Convolutional neural net:** specialized neural Nets
- explicit assumption that inputs are images (reduces parameters)
  - regular neural net don't scale well w/ images bec maintain full connectivity for all neurons => many parameters and overfitting
- ConvNets have layers w/ neurons called activation volumes arranged in 3 dimensions (**length**, **height**, **depth**)
- neurons in layer only connected w/ small region of prev layer (not fully connected)
- for classification, full image reduced into a single vector of class scores

- ConvNet made of layers
- each layer transforms an input 3D volume to an output 3D volume w/ some differentiable function that may/may not have parameters


## ConvNet Layers
- use 3 main types of layers for ConvNet
1. Convolutional Layer
2. Pooling Layer
3. Fully-Connected Layer

- ex. ConvNet for CIFAR-10 classification [INPUT -> CONV -> RELU -> POOL -> FC]
1. INPUT [32x32x3]: hold raw pixel values of image (32x32 RGB)
2. CONV: computes output of neurons connected to input w/ dot product betw their weights and small region
- may result in volume 32x32x12 if using 12 filters
3. RELU: apply elementwise activation funct, ex **max(0,x)** thresholding at zero
- size still 32x32x12
4. POOL: downsamples along width & height (depth unchanged)
- size now 16x16x12
5. FC: computes class scores
- size now 1x1x10 (each # is a class score)

- ConvNets transform image layer by layer form original pixels to final classes
- not all layers have parameters
  - CONV/FC perform transformations that are funct of activations in input volume and params (weights & biases of neurons)
    - params trained w gradient descent so class scores that ConvNet computes are consistent w/ training labels
  - RELU/POOL layers use fixed funct
- some layers have additional hyperparameters
    - CONV/FC/POOL have hyperparameters, RELU doesn't


## Convolutional Layer (CONV)
- CONV layer accepts volume of size W1 X H1 X D1
- req 4 hyperparams
1. \# of filters, K
2. spatial extent F
3. stride S (distance betw centres of filters)
4. zero padding P
- produces a volume of size W2 X H2 X D2 where
W2 = (W1 - F + 2P)/S + 1
H2 = (H1 - F + 2P)/S + 1 (height & weight computed equally by symmetry)
D2 = K
- w/ param sharing, it introduces F*F*D1 weights per filter for a total of (F*F*D1)* K weights and K biases
- in ouput volume, d-th slice (W2*H2) is result of convolution of dth filter over input volume w/ stride S and then offset by dth biases
- common setting of hyperparams: F=3, S=1, P=1

- filters are small spatially (along width & height), but goes full depth
  - ex. 5x5X3 filter (5x5 RGB)
- during forward pass, filter convolved across width and height of input volume
  - computes dot product betw entries of filter and input at any position
  - produces a 2D activation map that gives responses of filter at every spatial position
- ConvNet learns filters that activate when they see some type of visual feature (ex. edge, blotch of colour)
- stack activation maps along depth dimension and produce output volumes

**receptive field (filter)** hyperparam rep spatial extent of connectivity of each neuron to a local region of input volume
- extent of connectivity along depth axis always equals depth of input volumes
- Note: asymmetry in spatial dimens (width & height) and depth dimens
  - connections are local in space (width & height) but always full along entire depth

- ex. input volume [32x32x3] w/ filter [5x5] => each neuron in CONV have weights to [5x5x3] region in input volume w/ total 5*5*3 = 75 weights (+1 bias param). Connectivity along depth axis is 3 (depth of input volume)


### Spatial Arrangement
- 3 hyperparams control size of output volume
1. depth
2. stride
3. zero-padding

**depth:** corresponds to # of filters
- each filter learns to look for something diff in the input
**depth column:** set of neurons that are all looking at the same region of the input

**stride:** speed at which the filter is moved
- stride = 1 => move filters 1 pixel at a time
- larger strides produce smaller output volumes spatially

**zero-padding**: padding of zeros for the input volume
- allows us to control the spatial size of the output volumes
- usually use to exactly preserve the spatial size of the input volume

**spatial size of output volume:**

Size(W,F,S,P) where W is input volume size, F is the receptive field size of CONV layer, S is the stride, P is the amt of zero padding on border

Size(W,F,S,P) = (W - F + 2P)/S + 1

Note: stride is constrained by W, F, P since (W-F+2P)/S must be an integer
- otherwise, neurons don't fit symmetrically across input
- considered invlaid setting of hyperparmas and ConvNet library could throw exception or zero pad or crop to make it fit

### Parameter Sharing
- param sharing used in CONV to control # of parameters
- can reduce # of params by assuming if a feature is useful to compute at some spatial posiiton (x,y), then it should also be useful to compute it at a diff position (x2, y2)
- i.e. denote a single 2D slice of depth as a **depth slice** will constrain the neurons in each depth slice to use the same weights and biases
- thus CONV layer would have only 1 weight per depth slice
- during backpropagation, every neuron in the volume will compute the gradient for its weights, but these gradients will be added up across each depth slice and only update a single set of weights per slice
- ex. a [55x55x96] input has 96 depth slices (55x55) and uses only 96 unique weights

- Note: if all neurons in a single depth slice using hte same weight vector, the forward pass of CONV layer in each depth slice can be computed as a **convolution** of neurons weights w/ input volume
- usual refer to the sets of weights as a **filter** or **kernel** that is convolved w/ input

- param sharing does not work when input images to ConvNet has a specific centered structure where we expect completely diff features should be learned on each side
  - ex. if input are faces centered in image, eye-specifc or hair-specific features should be learned in diff spatial locations
- can relax param sharing scheme and call the layer a **locally-connected layer**

### Numpy examples
input volume: numpy array X
depth column at position (x,y): X[x,y,:]
depth slice/activation map @ depth d: X[:,:d]

-ex input volume X has shape X.shape: (11,11,14) and use zero padding (P=0), filter size F=5, and stride is S=2
- output volume has size: (11-5)/2 +1 = 4 giving volume width & height of 4
- activation map in output volume V:

V[0,0,0] = np.sum(X[:5,:5,:]) * W0) + b0
V[1,0,0] = np.sum(X[2:7,:5,:] * W0) + b0
V[2,0,0] = np.sum(X[4:9,:5,:] * W0) + b0
V[3,0,0] = np.sum(X[6:11,:5,:] * W0) + b0

Note: * in numpy means elementwise multiplication
- W0 is the weight vector of the neuron and b0 is the bias
- assume w0 is of shape W0.shape: (5,5,4)
- at each pt compute dot product and use the same weight & bias (param sharing)
- dimenions along width are increasing in steps of 2 (stride)

- 2nd activation map in V:

V[0,0,1] = np.sum(X[:5,:5,:] * W1) + b1
V[1,0,1] = np.sum(X[2:7,:5,:] * W1) + b1
V[2,0,1] = np.sum(X[4:9,:5,:] * W1) + b1
V[3,0,1] = np.sum(X[6:11,:5,:] * W1) + b1
V[0,1,1] = np.sum(X[:5,2:7,:] * W1) + b1 (example of going along y)
V[2,3,1] = np.sum(X[4:9,6:11,:] * W1) + b1 (or along both)

### Implementation as Matrix Multiplication
- can treat convolution operation as big matrix multiply of filters and local regions of input
1. local regions in input image stretched into columns using **im2col**
- ex. for input [227x227x3] convolved w/ 11x11x3 filters at stride 4, then take 11x11x3 blocks of pixels in inputs and stretch block into 11*11*3=363 column vector. Iterating process at stride 4 gives (227-11)/4 +1 =  55 locations along width and height, leading to output matrix X_col of im2col of size 363x3025 where every column is a stretched out receptive field/filter
2. weights of CONV layer are similar stretched into rows
- ex. if 96 filters of size 11x11x3 => matrix `W_row` of size 96x363
3. result of convolution is now equivalent to np.dot(W_row, X_col) which evaluates dot product betw every filter and receptive field location
- ex. output is 96x3025 matrix
4. result must finally be reshaped into proper output dimens 55x5x96

**backpropagation:** backward pass for convolution operation (both data & weights) is also a convolution (but w/ spatially-flipped filters)
**dilated convolutions:** use filters w/ spaces betw each cell (dilation)
- dilation is a hyperparam
- ex. dilation = 0 => `w[0]*x[0] + w[1]*x[1] + w[2]*x[2]`
dilation = 1 =>`w[0]*x[0] + w[1]*x[2] + w[2]*x[4]`
- can use dilation=1 w/ dilation=0 to merge spatial info across inputs w/ fewer layers
- ex. if you stack two 3x3 on top of each other, the neurons on 2nd layer are a funct of a 5x5 patch of input (effective receptive field is 5x5)


# Pooling Layer
- usually insert Pooling layer betw successive CONV Layers
- reduces spatial size of representation to reduce amt of params and computation in network and thus control overfitting
- operates indep on every depth slice and resizes it using MAX operation
- usually use filters of size 2x2 applied w/ stride 2 downsamples/depth slice in put by 2 along height & width
  - takes max of 4 numbers
  - reduces size to 1/4 of original (depth stays constant)

- pooling layer accepts volume of size W1xH1xD1
- req 2 hyperparams
1. spatial extent (F)
2. stride (S)
- produces a volume of size W2*H2*D2 where
  - W2=(W1-F)/S + 1
  - H2=(H1-F)/S + 1
  - D2=D1
- introduces 0 params since it computes a fixed funct of inputs
- usually do not zero pad
- 2 common pooling layers
1. F=2, S=2  
2. F=3, S=2 (overlapping pooling)
- pooling sizes w/ larger receptive fields are too destructive

### General Pooling
- pooling units can be used for max pooling, average pooling, or L2-norm pooling
- historically used average, but now use max pooling

**backpropagation:** backward pass for a max(x,y) only route sthe gradient to the input that had the highest value in the forward pass
- usually track index of max activation (switches) in forward pass of pooling layer to keep gradient routing efficient during backpropagation
- some people avoid using pooling and use repeated CONV layers instead
  - to reduce size of rep, use larger stride in CONV layer once in a while
- discarding pooling layers impornt for good generative models (ex. variational autoencoders, generative adversarial nerworks)

## Fully Connected layers
- neurons in FC have full connections to all activations in prev layer
- can compute activatios w/ matrix multiplication w/ bias offset

### Converting FC Layers to CONV Layers
- only diff is neurons in CONV layer connected only to local region in input and many neurons share params
- same functional form (dot products)
- for any CONV layer, there is an FC layer w. same forward function
  - weight matrix is large matrix that is mostly zero except at certian blocks (bec local connectivity) and where weights in many of the blocks are equal (bec of param sharing)
- can express any FC as CONV layer
  - ex. for FC layer K=4096 for input volume 7x7x512, equivalent CONV is F=7, P=0, S=1, K=4096
  - i.e set filter size to be exaclty size of input volume => output is 1x1x4096 since only a single depth column fits across the input volume

**FC->CONV conversion:** useful to convert FC to CONV bec can slide original ConvNet efficienlty across many spatial positions in a larger image in a single forward pass
- for better performance, can resize an image to make it bigger, use a converted ConvNet to evaluate class scores at amny spatial positons and then average class scores


## ConvNet Architechtures
**RELU:** activation function which applies elementwise non-linearity

- most common form of ConvNet:
  `INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`
  where N>=0 (usually N<=3), M>=0, K>=0 (usually K<3)

examples:
- `INPUT -> FC`: linear classifier (N=M=K=0)
- `INPUT -> CONV -> RELU -> FC`
- `INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC`
  - single CONV layer betw every pool
- `INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC`
  - 2 CONV layers before every POOL
  - usually good for larger and deeper networks bec multiple stacked CONV layers can develop more complex features of input volume before destructive pooling operation

- prefer stack of small filter CONV to one larger receptive CONV layer
  - allows expressiong of more powerful features of input w/ fewer parameters but req more memory to hold all intermediate CONV layer results if backprop
  - ex. using three 3x3 CONV layers on top (w/ non-linearities) vs  sinlge CONV w/ 7x7 receptive field. Effective receptive field is 7x7, but neurons in single 7x7 are computing a linear funct over input, while 3 stacks of CONV contain non-linearities. Also if all volumes have C channels, then single 7x7 CONV has C*(7x7xC)=49C^2 channels while three 3x3 CONV have 3x(Cx(3x3xC))=27C^2 channels


## Layer Sizing Patterns
1. **INPUT layer:** should be divisible by 2 many times  
- ex 32 (CIFAR-10), 64, 96 (STL-10), 224 (ImageNet), 384, 512
2. **CONV layers:** small filters (3x3 or 5x5 at most) w/ S=1 and padding input volumes w/ zeros s.t. CONV layer does not alter spatial dimensions of input
3. **POOL layers:** usually max-pooling w/ 2x2 receptive fields (F=2) w/ stride = 2
- discards exactly 755 of activations in input volumes
- receptive fields > 3x3 are too lossy and aggressive => worse performance

- this pattern only does downsampling in POOL layers
  - easier to track than downsampling in CONV and POOL
- smaller strides work better
- using padding prevents info at borders from being washed away too quickly
   - no padding causes sizes of volume to reduce by small amts each time


## Famous Nets   
1. **LeNet**
- 1st successful applications of ConvNets (1990s)
- single CONV layer immediately followed by POOL layer
2. **AlexNet**
- 1st work popularizing ConvNets in Computer Vision (2012)
- similar to LeNet but was deeper, bigger, and used Conv layers stacked on top of each other
3. **ZF Net**
- improved AlexNet by tweaking architecture hyperparams
- expanded size of middle CONV layers and made stride & filter size on 1st layers smaller
4. **GoogLeNet**
- developed an Inception Module that reduced \# of params in network (4M vs AlexNet 60M)
- uses Average Pooling instead of FC at to of ConvNet
  - eliminated insignificant params
5. **VGGNet**
- showed depth of network is critical for good performance
- used 16 CONV/FC layers
- homogenous architecture that only performs 3x3 convolutions and 2x3 pooling
- more expensive to evaluate & uses a lot more memory and params (140M)
- most params in 1st FC, but later found that these FC layers can be removed w/ no performance downgrade
6. **ResNet (Residual Network)**
- uses special skip connections and lots of batch normalization
- missing FC layers at end of network   


## Computational Considerations
- largest bottleneck for ConvNet is memory bottleneck
- modern GPUs have aroudn 3-6 GB memory w/ top having 12 GB
- 3 major sources of memory usage
1. Intermediate volume sizes
- most **activations** in earlier CONV layers of ConvNet
- kept bec needed for backprop
- but if only running ConvNet at test time can reduce activations by only storing current activ at any layer and discarding the prev activations on lower layers
2. Param sizes
- \#'s that hold network **parameters**, gradients during backprop, and commonly a step cache if optimization is using momentum, Adagrad, or RMSProp
- memory to store param vector alone usually mult by at least factor of 3
3. **Miscellaneous** memory
- image data batches, augmented versions, etc

- once have rough estimate for total \#'s of values (for activations, gridents, and misc), convert size to GB
- multiple \# of values by 4 to get raw bytes (every floating pt is 4 bytes, 8 for double precision) and then divide by 1024 to get amt of memory in KB, MB, and GB
- if network doesn't fit, make it fit by decreasing batch size since most memory consumed by activations
