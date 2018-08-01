# Neural Networks

# Perceptron Trick
Consider a linear classifier and a set of points. Training our classifier shall find a line that separates the data. We will now learn a trick that modifies the equation of a line so that it comes closer to a particular point.

Let's consider the line $3x_{1} + 4x_{2} - 10 = 0$ and a point $(4,5)$.

We can now move to line closer to the point by doing the following:

<img src="images/perceptron_trick.png" width="350"/>

The new line will be: $-1x_{1} - 1x_{2} - 10 = 0$ <br/>
However, doing this will move the line drastically towards the point and eventually misclassify many other points. To overcome this problem we want to make small steps towards this point. This can be done by the introduction of a learning rate.

<img src="images/perceptron_trick_learning_rate.png" width="350"/>


# Perceptron Algorithm

<img src="images/perceptron_algorithm.png" width="350"/>

# Sigmoid Activation Function

In order to use Gradient Descent we need to have a continuous error function. In order to do this, we also need to move from discrete predictions to continuous predictions. <br/>

Therefore, we will replace the step function by the sigmoid function.

$Sigmoid(x) = \frac{1}{1+e^{-x}}$

# Maximum Likelihood

Let's assume we have two models. One that tells me that the probability of a given label is 0.8% and another one that tells me that the probability is 0.55%. The question which model is more accurate given a series of events.

The best (most accurate) model is the model that gives us the highest probability that an event happened to us. This is called "Maximum Likelihood".

But how can we maximize such a probability? <br/>
Well, first we calculate the probabilities of a given point and multiply them together. However, the product of those probabilities might become very small and we might have problems to deal with it. <br/>
One way to overcome this problem is the conversion of product into sums by applying the logarithm.

# Cross Entropy

When taking the logarithm of a number between 0-1 the logarithm will be a negative number. Therefore, it makes sense to consider the negative of the logarithm to get positive numbers. This is why Cross Entropy is often called _negative log-likelihood_.

<img src="images/cross_entropy.png" width="350"/>

**Two classes:** <br/>
$\text{Cross Entropy} = - \sum{y_{i} \cdot ln(p_{i})+(1-y_{i}) \cdot ln(1-p_{i})}$

**Multiple classes:** <br/>
$\text{Cross Entropy} = - \sum{\sum{ y_{ij} \cdot ln(p_{ij})}}$


# Logistic Regression

One way to compute the error function is by using the following formula:

$\text{Error Function} = -\frac{1}{m}{\sum{ (1-y_{i}) \cdot ln(1-y_{i}) + y_{i} \cdot ln(p_{ij})}}$

**Note:**

- If the label should be classified "1": $y_{i}$ = 1 and therefore the first term becomes 0.
- If the label should be classified "0": $y_{i}$ = 0 and therefore the second term becomes 0.

# Perceptron vs Gradient Descent

<img src="images/perceptron_vs_gradient_descent.png" width="400"/>

As we can we can see Perceptron Algorithm is actually the same as Gradient Descent. The only difference is that in Perceptron Learning we $\widehat{y}$ can only be 1 or 0.