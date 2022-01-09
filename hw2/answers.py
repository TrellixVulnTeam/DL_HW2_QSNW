r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 5
    hidden_dims = 160
    activation = "relu"
    out_activation = "logsoftmax"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part1_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.01
    weight_decay = 0.007
    momentum = 0.25
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part1_q1 = r"""
**Your answer:**

Optimization error - We do have an optimization error since we
get into a local minimum with the train set which is not globally
(we don't get 100% accuracy) but we don't think it is very high since
the accuracy is still high.

Generalization error - We see in the graphs that the train batch is 
improving rapidly but the test batch is not stable.
This is because we have a relatively small data set and it means that we 
have a lot of noise in our data we train on. So, the generalization error
is high.

Approximation error - We used here MLP with 1 activation function
and 1 out activation function. By definition from the tutorial we use
a limited set of possible functions and therefore we have an high approximation
error.

"""

part1_q2 = r"""
**Your answer:**

We can see that in the creation of the data the validation data set
has more orange dots (class 1 which is the default positive class in the
Binary Classifier) in the blue area than blue dots in the orange
area, therefore we suspect that FNR will be higher than FPR.

"""

part1_q3 = r"""
**Your answer:**

In most cases the "optimal" ROC point is not relevant. For each case we
will decide what is more important to decrease, FNR or FPR.

1. In this case since the symptoms are not lethal and the patient could be
immediately treated FNR is not so bad, but if we get FPR than the patient will
do expensive tests. In this case we will aim to a higher FNR than FPR.

2. In this case we must get the lowest FNR possible because patient could
die with high probability if not diagnosed early and correctly.
In this case we will aim to a higher FPR than FNR.

"""


part1_q4 = r"""
**Your answer:**

1. We can see in each column a fixed depth with changing width.
We see that when the width is very small (for example, 2) the model is too
simple and we get bad results. When the width is very high (for example, 128)
the model is too complicated. Is seems that the best results are somewhere in
the middle (8 or 32).

2. We can see in each row a fixed width with changing depth.
In the first row when the width is 2 which is very small the best depth is 4.
In the other rows when the width is bigger than the best depth is 2.
The reason could be that when the width is to small than the model is simple
and we need more depth to make it a little more complicated and accurate.
But, when the width is big enough too much depth could get the model over
fitted with our data.

3. In the first case we see that 4x8 is more accurate than 1x32. The reason
could be that in 1x32 the width is large but the depth is too small to be accurate
enough and more depth we smaller width is better.

In the second case, the results are almost the same. We see that 1x128 is worst
because it has only 1 depth and big width, and 4x32 is more balanced which
create better accuracy. 

4. Our model can be over fitted to the data and because of that the threshold
is very important. Using tuned threshold can make make the model more generalized
and we see that most of the time we get better results after tuning it.

"""
# ==============
# Part 2 answers


def part2_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()

    lr = 0.01
    weight_decay = 0.007
    momentum = 0.83
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part2_q1 = r"""
__________________________________________________________________________________
1) NUMBER OF PARAMETERS:

Formula:
ParamSet = (kernelH*kernelW*inChannel + bias)*(outChannels)


Bottleneck of parameters (3 Layers):

Param Set1 = (1*1*256+1)*64 = 16,448
Param Set2 = (3*3*64+1)*64 = 36,928
Param Set3 = (1*1*64+1)*256 = 16,640
Total Parameters: 70,016


Normal Number of parameters (2 Layers):

Param Set1 = (3*3*256+1) = 590,080
Param Set2 = Param Set1 = 590,080
Total Parameters: 1,180,160
__________________________________________________________________________________
2) FLOATING POINT OPERATIONS:

Formulas:
Relu Floating Point Operations = outChannels*HW
Convolution Floating Point Operations = (kernelH*kernelW*in_channel+bias)*outChannel*HW
Residual = outChannel
Where HW = Hout*Wout


Bottleneck number of operations (3 Layers):

Convolution: (1*1*256+1)*64*HW = 16,448HW
Activation (RELU): 64HW
Convolution: (3*3*64+1)*64HW = 36,864HW
Activation (RELU): 64HW
Convolution: (3*3*64+1)*256HW = 147,456HW
Residual = 256HW
Number of operations: 201,152HW


Normal number of operations (2 Layers):

Convolution: (3*3*256+1)*256HW = 590,080HW
Activation (RELU): 256HW
Convolution: (3*3*256+1)256HW = 590,080HW
Residual = 256HW
Number of operations: 1,180,672HW
__________________________________________________________________________________
3) FLOATING POINT OPERATIONS:
Bottleneck: Combining features.
Normal: Feature maps not reduced. No feature combinations
__________________________________________________________________________________

Conclusion:

            Bottleneck                              Normal
Parameters: 70,016                    |  Parameters: 1,180,160
Easy to tune small parameter amounts  |  Tons of parameters, hard to tune
                                      |
Bottleneck of operations: 201,152HW   |  Normal number of operations: 1,180,672HW
Easier to compute                     |  Long execution time, hard to tune parameters with such long runtimes.
                                      |
Reducing features allows us to reduce |  No feature reduction
computation amounts and RAM           |

__________________________________________________________________________________
"""

# ==============

# ==============
# Part 3 answers


part3_q1 = r"""
1. As we build more intricate, deeper models we gain an increase in the accuracy rate also increases and enables the
extraction of more accurate and intricate features, when we go too deep, our accuracy starts to diminish.
The best results are for depth=6 with above 70% accuracy, and we speculate that we may not have enough data for the 
deeper model to extract the features correctly and take full advantage of its power.

2. The model could not train on K=64, L=16. Since the loss graph is a flat line, we speculate that this could be the
result of vanishing gradients. result of vanishing gradients. Two options to try and improve the results is the use
of batchnorms or residual networks. 
"""

part3_q2 = r"""
As we increase the number of filters, we ger increasingly better accuracy (also when comparing to experiment 1.1).
We assume that the high number of filters assist us in creating good high-quality
features that assist our models to classify the samples better.
"""

part3_q3 = r"""
Once again, we can see the issue of the vanishing gradient showing its face for the L=4, K = [64, 128,256].
We can also see that comparing with the previous experiments, our results do not get alot better. Another interesting
point is that we can see spiking in the loss on large amounts of iterations while using small L-s on the
training set which suggests overfitting.
"""

part3_q4 = r"""
Our best results for the K=32 are for L=8, and L=2 for K= [64 128 256]. We do have to be carefull with the latter, since
we can see the loss graph spike up towards the most recent iterations on the test set, suggesting overfitting.
"""

part3_q5 = r"""
1. Using ResNet CNN we attempted to reduce the overfitting many of the previous models had. We added batchnorm in order
 to resolve the issue, and we also tested different activation functions and tried to optimize our parameters.
 Eventually we decided to go with tanh, and added dropout and batchnorm, optimizing using the checkpoint and early stopping
 we implemented in previous ex.

2. Comparing to previous attempts, we can see an improvement with ~77% accuracy. Also we attempted to minimise the loss
by using the new functions we implemented and what we learned from running the previous exercises.
"""
# ==============
