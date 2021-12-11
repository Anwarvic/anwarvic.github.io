---
title: "wav2vec 2.0"
date: 2020-10-22
cover: /image1.png
---

Wav2Vec Unsupervised is a model created by Facebook AI Research in May
2021 and published in this paper: [wav2vec 2.0: A Framework for
Self-Supervised Learning of Speech
Representations](https://arxiv.org/pdf/2006.11477.pdf). wav2vec 2.0
leverages self-supervised training, like vq-wav2vec, but in a continuous
framework from raw audio data.

Inspired by the end-to-end version of the vq-wav2vec paper, the authors
further explored this idea with a novel model architecture that consists
of the following modules:

-   **Feature Encoder**: Takes raw audio
    $\mathcal{X} = x_{1},\ x_{2},\ ...\ x_{T}$ and outputs latent
    speech representation $\mathcal{Z} = z_{1},\ z_{2},\ ...z_{T}$ for
    $T$ time-steps.

-   **Transformer**: Takes latent representations
    $\mathcal{Z} = z_{1},\ z_{2},\ ...z_{T}$ and outputs context
    representations $\mathcal{C} = c_{1},\ c_{2},\ ...c_{T}$.

-   **Quantization Module**: Takes latent representations
    $\mathcal{Z} = z_{1},\ z_{2},\ ...z_{T}$ and outputs quantized
    vectors $\mathcal{Q} = q_{1},\ q_{2},\ ...q_{T}$.

<div align="center">
    <img src="media/wav2vec_2/image1.png" width=750>
</div>

As we can see, this model has the same idea as the earlier paper with a
few differences:

-   wav2vec 2.0 builds context representations over continuous speech
    representations while vq-wav2vec uses discrete speech
    representations.

-   wav2vec 2.0 uses transformers in their architecture whose
    self-attention captures dependencies over the entire sequence of
    latent representations end-to-end which is different from
    v1-wav2vec.

-   wav2vec 2.0 outperforms vq-wav2vec even when using 10 times less
    labeled data than vq-wav2vec.

Now, let's talk in a little bit more details about the wav2vec 2.0
model:

Feature Encoder
---------------

As we said earlier, the feature encoder takes normalized raw audio
$\mathcal{X} = x_{1},\ x_{2},\ ...\ x_{T}$ and outputs latent speech
representation $\mathcal{Z} = z_{1},\ z_{2},\ ...z_{T}$ for $T$
time-steps. The raw waveform input to the encoder is normalized to zero
mean and unit variance.

The feature encoder consists of seven layers of temporal convolution
with strides (5,2,2,2,2,2,2) and kernel widths (10,3,3,3,3,2,2). Each
layer has 512 channels followed by a layer normalization and a GELU
activation function. The stride determines the number of time-steps T
returns from the encoder which are the input to the Transformer.

Context Transformers
--------------------

The output of the feature encoder is fed to a context network which
follows the Transformer architecture. Original architecture of
transformers have positional embedding layer. In this architecture, they
used a convolution layer acting as positional embedding. This
convolution layer has a kernel size of 128 and 16 channels. Then, they
followed the output of the convolution by a GELU and layer
normalization.

In the paper, they experimented with two model configurations:

-   <u><strong>BASE:</strong></u>\
    It contains 12 transformer blocks, model dimension 768,
    inner dimension (FFN) 3,072 and 8 attention heads and dropout of
    0.05. We optimize with Adam, warming up the learning rate for the
    first 8% of updates to a peak of $5 \times 10^{- 4}$ and then
    linearly decay it.

-   <u><strong>LARGE:</strong></u>\
    It contains 24 transformer blocks with model dimension
    1,024, inner dimension 4,096 and 16 attention heads and dropout of
    0.2. We optimize with Adam, warming up the learning rate for the
    first 8% of updates to a peak of $3 \times 10^{- 4}$ and then
    linearly decay it..

Quantization Module
-------------------

For self-supervised training we discretize the output of the feature
encoder $\mathcal{Z}$ to a finite set of speech representations
$\mathcal{Q}$ via <u><strong>product quantization</strong></u>. Product
quantization amounts to choosing quantized representations from multiple
codebooks and concatenating them.

<div align="center">
    <img src="media/wav2vec_2/image2.png" width=750>
</div>

Product quantization is done in the following steps:

-   The feature encoder output $\mathcal{Z}$ is inserted to $G$ linear
    layers; each followed by a ReLU, followed by another linear which
    outputs logits $l \in R^{G \times V}$ as shown in the following
    formula:

$$l = g\left( \text{ReLU}\left( f\left( \mathcal{Z} \right) \right) \right)$$

-   Then, we get the probability out of these logits
    $p \in R^{G \times V}$. The following formula shows the
    probability for choosing the j^th^ variable of the g^th^ group
    where $u$ is a vector of uniform sampled values from
    $\mathcal{U}\left( 0,\ 1 \right)$ and $\mathcal{t}$ is the
    temperature which is a non-negative hyper-parameter:

$$p_{g,j} = \frac{\frac{\exp\left( l_{g,j} + v_{j} \right)}{\mathcal{t}}}{\sum_{k = 1}^{V}\left( \frac{\exp\left( l_{g,k} + v_{k} \right)}{\mathcal{t}} \right)}$$

$$v = - \log\left( - \log\left( u \right) \right)$$

-   We choose one entry from each codebook and concatenate the resulting
    vectors $e_{1},\ ...\ e_{G}$.

$$i = \text{argma}x_{j}\left( p_{g,i} \right)$$

-   Then, we apply a linear transformation to obtain $q \in \mathbb{R}^{f}$.

Training
--------

The training objective requires identifying the correct quantized latent
audio representation in a set of distractors for each masked time step
the same as the earlier paper. In this paper, we are going to do that by
masking a proportion of the feature encoder outputs, or time steps
before feeding them to the context network (transformer) unlike the
vq-wav2vec paper which didn't do masking.

We randomly sample without replacement a certain proportion p=0.065 of
all time steps to be starting indices and then mask the subsequent M=10
consecutive time steps from every sampled index; spans may overlap.

The loss function for this training can be described in the following
formula:

$$\mathcal{L} = \mathcal{L}_{m} + \alpha\mathcal{L}_{d}$$

As we can see, the loss function consists of two terms with a
hyper-parameter $\alpha$ to determine the weight of each term:

-   **Contrastive Loss $\mathcal{L}_{m}$:**\
    Given context network output <span>$c_{t}$</span> centered over masked time
    step <span>$t$</span>, the model needs to identify the true quantized latent
    speech representation <span>$q_{t}$</span> in a set of <span>$K + 1$</span>
    quantized candidate representations <span>$\widetilde{q} \in \mathcal{Q}_{t}$</span>
    which includes <span>$q_{t}$</span> and <span>$K$</span> distractors.
    Distractors are uniformly sampled from other masked time steps of the same
    utterance. The loss is defined as:

$$\mathcal{L}_{m} = - log\left( \frac{\exp\left( \frac{\text{sim}\left( c_{t},q_{t} \right)}{k} \right)}{\sum_{\widetilde{q}\sim\mathcal{Q}}^{}{\exp\left( \frac{\text{sim}\left( c_{t},q_{t} \right)}{k} \right)}} \right),\ \ sim\left( a,b \right) = \frac{a^{T}\text{.b}}{\left\| a \right\|.\left\| b \right\|}$$

-   **Diversity Loss $\mathcal{L}_{d}$**\
    The diversity loss is designed to increase the use of the
    quantized codebook representations by encouraging the equal use of
    the $V$ entries in each of the $G$ codebooks:

$$\mathcal{L}_{d} = \frac{1}{\text{GV}}\sum_{g = 1}^{G}{- H\left( p_{g} \right)} = \frac{1}{\text{GV}}\sum_{g = 1}^{G}{\sum_{j = 1}^{V}{p_{g,j}\text{.log}\left( p_{g,j} \right)}}$$
