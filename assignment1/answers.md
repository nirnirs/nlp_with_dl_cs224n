# Assignment #1

##  1 Softmax
(a) Prove that softmax is invariant to constant offsets in the input.

$$\text{softmax}(x)=\text{softmax}(x+c)$$

#### Answer:
$$\text{softmax}(x+c)_i
=\frac{e^{x_i+c}}{\sum _{j=1}^n{e^{x_j+c}}}
=\frac{e^{x_i} \cdot e^c}{e^c \cdot \sum _{j=1}^n{e^{x_j}}}
=\frac{e^{x_i}}{\sum _{j=1}^n{e^{x_j}}}
=\text{softmax(x)}_i$$

##  2 Neural Network Basics

(a) Derive the gradient of the sigmoid function, and show that it can be rewritten as a function of the function value.

#### Answer:
$$\sigma'(x) 
=\left(\frac{1}{1+e^{-x}}\right)'
=-\frac{1}{(1+e^{-x})^2} \cdot -e^{-x}
=\frac{e^{-x}}{(1+e^{-x})^2}
=\frac{e^{-x}}{1+e^{-x}} \cdot (1 - \frac{e^{-x}}{1+e^{-x}} )
=\sigma(x) \cdot \left(1 - \sigma(x) \right)$$

(b) Derive the gradient with regard to the inputs of a softmax function when cross entropy loss is used for evaluation.

#### Answer:
$$\text{CE}(y,\hat y)
=-\sum _i{y_i}\log{\hat {y_i}}
=-\log{\hat {y_k}}
=-\log({\text{softmax}(\theta)}_k)
=\log\left(\frac{\sum _{i=1}^n{e^{\theta _j}}}{e^{\theta _k}}\right)
=\log\left(\sum _{i=1}^n{e^{\theta _j}}\right) - \theta _k$$

where $k$ is the index of the correct class. Then:

$$\frac{\partial \text{CE}(y,\hat y)}{\partial \theta{_k}}
=\frac{\partial}{\partial \theta{_k}} \left(\log\left(\sum _{j=1}^n{e^{\theta _j}}\right) - \theta _k \right)
=\frac{e^{\theta _k}}{\sum _{j=1}^n{e^{\theta _j}}}-1
=\hat y_k - 1
$$

$$\frac{\partial \text{CE}(y,\hat y)}{\partial \theta_{i \neq k }}
=\frac{\partial}{\partial \theta_{i \neq k}} \left(\log\left(\sum _{j=1}^n{e^{\theta _j}}\right) - \theta _k \right)
=\frac{e^{\theta _i}}{\sum _{j=1}^n{e^{\theta _j}}}
=\hat y_i
$$

or, equivalently:
$$\frac{\partial \text{CE}(y,\hat y)}{\partial \theta}=\hat y - y$$


(c) Derive the gradients with respect to the inputs x to an one-hidden-layer neural network.

Let: 
$$z_1=xW_1+b_1$$
$$z_2=hW_2+b_2$$

Then:
$$h=\sigma(z_1)$$
$$\hat{y}=\mathrm{softmax}(z_2)$$

The gradient with respect to $x$ is:
$$\frac{\partial{\mathrm{CE}(y, \hat y)}}{\partial{x}}
=\frac{\partial{z_1}}{\partial{x}}
\cdot \frac{\partial{h}}{\partial{z_1}}
\cdot \frac{\partial{z_2}}{\partial{h}}
\cdot \frac{\partial{\mathrm{CE}(y, \hat y)}}{\partial{z_2}}
$$

where:

$$\frac{\partial{z_1}}{\partial{x}} = W_1$$

$$\frac{\partial{h}}{\partial{z_1}} = \sigma'(z_1)$$

$$\frac{\partial{z_2}}{\partial{h}} = W_2$$

$$\frac{\partial{\mathrm{CE}(y, \hat y)}}{\partial{z_2}}  = \hat{y} - y$$

and so:

$$\frac{\partial{\mathrm{CE}(y, \hat y)}}{\partial{x}}
=(\hat{y}-y) \cdot W_2^T \cdot \sigma'(z_1) \circ W_1^T $$

Where $\circ$ means broadcasting.

(d) How many parameters are there in this neural network, assuming the input is $D_x$-dimensional, the output is $D_y$-dimensional, and there are $H$ hidden units?

$W_1$ is of size $D_x \cdot H$, and $W_2$ is of size $H \cdot D_y$ so the total number of parameters in the network is:
$$W_1 + W_2 = D_x \cdot H +  H \cdot D_y = H \cdot (D_x + D_y)$$

##  3 word2vec

(a) Derive the gradients of a word2vec skipgram with respect to $v_c$.

#### Answer:
$$\frac{ \partial{CE(y, \hat{y})}}{ \partial{v_c}}
=\frac{ \partial{CE(y, \hat{y})}}{ \partial{v_cU}}
\cdot \frac{ \partial{v_cU}}{\partial{v_c}}
= (\hat y - y) \cdot U^T$$

(b) Derive the gradients of a word2vec skipgram with respect to $u_w$.

#### Answer:
$$\frac{ \partial{CE(y, \hat{y})}}{ \partial{v_c}}
=\frac{ \partial{CE(y, \hat{y})}}{ \partial{v_cU}}
\cdot \frac{ \partial{v_cU}}{\partial{U}}
=v_c^T \cdot (\hat y - y)$$