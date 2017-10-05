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

(a) Derive the gradients of a word2vec skipgram with softmax-crossentropy cost with respect to $v_c$.

#### Answer:
$$\frac{ \partial{CE(y, \hat{y})}}{ \partial{v_c}}
=\frac{ \partial{CE(y, \hat{y})}}{ \partial{v_cU}}
\cdot \frac{ \partial{v_cU}}{\partial{v_c}}
= (\hat y - y) \cdot U^T$$

(b) Derive the gradients of a word2vec skipgram with softmax-cross entropy cost with respect to $u_w$.

#### Answer:
$$\frac{ \partial{CE(y, \hat{y})}}{ \partial{U}}
=\frac{ \partial{CE(y, \hat{y})}}{ \partial{v_cU}}
\cdot \frac{ \partial{v_cU}}{\partial{U}}
=v_c^T \cdot (\hat y - y)$$

(c) Repeat parts (a) and (b) assuming we are using the negative sampling loss for the predicted vector $v_c$, and the expected output word is $o$.

#### Answer:

Define:
$$\mathrm{z}(u, v)=-\log(\sigma(u^Tv))$$

$\sigma(uv)$ will be close to 1 when $u$ and $v$ point in the same direction, and closer to 0 when $u$ and $v$ point in different directions.
$-\log\sigma(uv)$ will be higher when $u$ and $v$ point in different directions. So basically $\mathrm{z(u,v)}$ is asort of dissimilarity metric,
and so the cost is high when the output vector is not similar to the center vector, and when the center vector is similar to the negative sampled vectors.


$$\frac{\partial{\mathrm{z}(u, v)}}{\partial{v}})
=\frac{\partial{-\log\left(\sigma(u^Tv)\right)}}{\partial{v}}
=-\frac{1}{\sigma(u^Tv)} \cdot \sigma(u^Tv) \cdot \left(1 - \sigma(u^Tv)\right) \cdot u^T
=\left(1 - \sigma(u^Tv)\right) \cdot -u^T$$

Which basically means that in order to increase the dissimilarity we need to move $v$ in the opposite direction of $u$.

The gradient of $J$ with respect to $v_c$ is then:

$$\frac{ \partial{J(o,v_c,U)}}{\partial{v_c}}
=\frac{\partial}{\partial{v_c}} \left( \mathrm{z(u_o, v_c)} 
+\sum_{k=1}^K{\mathrm{z}(-u_k,v_c)}\right)
=\left(1 - \sigma(u_o^Tv_c) \right) \cdot -u_o^T
+\sum_{k=1}^K{\left(1 - \sigma(-u_k^Tv_c)\right) \cdot u_k^T}$$

Which basically means that in order to increase the cost, we need to move $v_c$ in the opposite direction of $u_o$ and in the same direction as $u_k$.

The gradient of $J$ with respect to $u_o$ is then:

$$\frac{ \partial{J(o,v_c,U)}}{\partial{u_o}}
=\frac{\partial}{\partial{u_o}} \left( \mathrm{z(u_o, v_c)}
+\sum_{k=1}^K{\mathrm{z}(-u_k,v_c)}\right)
=\left(1 - \sigma(u_o^Tv_c) \right) \cdot -v_c^T$$

Which basically means that the gradient of $J$ with respect to $u_o$ points in the opposite direction of $v_c$.

The gradient of $J$ with respect to $u_{w \in {1..k}}$ is then:

$$\frac{ \partial{J(o,v_c,U)}}{\partial{u_w}}
=\frac{\partial}{\partial{u_w}} \left( \mathrm{z(u_o, v_c)} 
+\sum_{k=1}^K{\mathrm{z}(-u_k,v_c)}\right)
=\left(1 - \sigma(-u_w^Tv_c)\right) \cdot v_c^T$$

Which basically means that the gradient of $J$ with respect to $u_w$ points in the same direction of $v_c$.

To compute the softmax-CE error, we need to compute the denominator of the softmax which requires to perform W dot products for the softmax denominator.
To compute the negative-sample error, we only need to perform K dot products. So the ratio is W/K.
