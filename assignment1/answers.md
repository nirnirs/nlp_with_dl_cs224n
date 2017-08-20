# Assignment #1

##  1 Softmax
(a) Prove that softmax is invariant to constant offsets in the input.

$$\text{softmax}(x)=\text{softmax}(x+c)$$

#### Answer:
$$\text{softmax}(x+c)_i 
=\frac{e^{x_i+c}}{\sum_{j=1}^n{e^{x_j+c}}}
=\frac{e^{x_i} \cdot e^c}{e^c \cdot \sum_{j=1}^n{e^{x_j}}}
=\frac{e^{x_i}}{\sum_{j=1}^n{e^{x_j}}}
=\text{softmax(x)}_i$$

##  2 Neural Network Basics

(a) Derive the gradient of the sigmoid function, and show that it can be rewritten as a function of the function value.

#### Answer:
$$\text{sigmoid}'(x) 
=\left(\frac{1}{1+e^{-x}}\right)'
=-\frac{1}{(1+e^{-x})^2} \cdot -e^{-x}
=\frac{e^{-x}}{(1+e^{-x})^2}
=\frac{e^{-x}}{1+e^{-x}} \cdot (1 - \frac{e^{-x}}{1+e^{-x}} )
=\text{sigmoid(x)} \cdot \left(1 - \text{sigmoid}(x) \right)$$

(b) Derive the gradient with regard to the inputs of a softmax function when cross entropy loss is used for evaluation.

#### Answer:
$$\text{CE}(y,\hat y)
=-\sum_i{y_i}\log{\hat {y_i}}
=-\log{\hat {y_k}}
=-\log({\text{softmax}(\theta)}_k)
=\log\left(\frac{\sum_{i=1}^n{e^{\theta _j}}}{e^{\theta _k}}\right)
=\log\left(\sum_{i=1}^n{e^{\theta _j}}\right) - \theta _k$$

where $k$ is the index of the correct class. Then:

$$\frac{\partial \text{CE}(y,\hat y)}{\partial \theta{_k}}
=\frac{\partial}{\partial \theta{_k}} \left(\log\left(\sum_{j=1}^n{e^{\theta _j}}\right) - \theta _k \right)
=\frac{e^{\theta _k}}{\sum_{j=1}^n{e^{\theta _j}}}-1
=\hat y_k - 1
$$

$$\frac{\partial \text{CE}(y,\hat y)}{\partial \theta_{i \neq k }}
=\frac{\partial}{\partial \theta_{i \neq k}} \left(\log\left(\sum_{j=1}^n{e^{\theta _j}}\right) - \theta _k \right)
=\frac{e^{\theta _i}}{\sum_{j=1}^n{e^{\theta _j}}}
=\hat y_i
$$

or, equivalently:
$$\frac{\partial \text{CE}(y,\hat y)}{\partial \theta}=\hat y - y$$