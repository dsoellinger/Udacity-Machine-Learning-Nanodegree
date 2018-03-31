# Support Vector Machines

## Idea
A normal classification algorithm usually tries to find a boundary (e.g. a line) that separates the data. However, the perfect position for such a line does often depend on the problem. Therefore, SVM introduce two additional boundaries parallel to the main line and tries to maximize this boundary. This margin should become as wide as possible.

<img src="images/svm.png" width="350"/>

## Errors
When trying to find the perfect boundary we differentiate between two different types of errors:

**Classification Error** <br/>
The data points which are wrongly classified.
When calculating the error function we weight the error based on the distance from the main boundary and sum them up.

<img src="images/error_function.png" width="350"/>

This is quite similar to the approach we know from the Perceptron Algorithm.
However, when calculating the error for SVM we don't start from the main line, instead we use our two additional boundaries as starting points for weighting since we also care about points inside the margin.

<img src="images/svm_classification_error.png" width="350"/>


**Margin Error** <br/>
Our goal is to obtain a model that has the largest margin as possible.
It can be shown that the distance between the two lines is always $\frac{2}{|W|}$. When then compute the Margin Error as follows: $|W|^2$

<img src="images/svm_margin_error_1.png" width="350"/>

<img src="images/svm_margin_error_2.png" width="350"/>


Finally, we compute the Total Error using the following formula:

$\text{Total Error} = \text{C} \cdot \text{Classification Error} + \text{Margin Error}$

The parameter C is simple constant that allows us to influence whether the model focuses more on correctly classifying points or on finding a large margin.

<img src="images/c_parameter.png" width="350"/>

