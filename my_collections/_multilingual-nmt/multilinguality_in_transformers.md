---
title: "Multilinguality in Transformers"
date: 2021-05-31
cover: /image0.png
labs: ["NAVER LABS Europe", "Université Grenoble Alpes"]
---

The following paper: [Do Multilingual Neural Machine Translation Models
Contain Language Pair Specific Attention
Heads?](https://arxiv.org/pdf/2105.14940.pdf) asks a very good question.
To answer it, the publishers tried to measure the importance of the
self-attention heads in the encoder and the encoder-decoder attention
heads of a many-to-one transformer. The NMT model was able to translate
French, German, Italian, Spanish, and Korean sentences to English. It
uses a variant of the Transformer-Big architecture with a shallower
decoder: 16 attention heads, 6 encoder layers, and 3 decoder layers on
[TED2020](https://opus.nlpl.eu/TED2020.php) dataset.

Denoting $\left| I \right|,\ \left| J \right|$ as the number of source
tokens and/or target tokens depending on whether we looked at the
self-attention of encoder or the encoder-decoder cross attentions, The
metrics used for importance are three:

-   **Confidence:**\
    It is the mean of its maximum attention weights.

$$\text{conf}\left( \text{head} \right) = \frac{1}{\left| I \right|}\sum_{i \in I}^{}{\max_{j \in J}\alpha_{i,j}}$$

-   **Variance:**\
    It's measured by how much each individual position $i$ is away from
    the expected position $\mu_{i}$:

$$\text{var}\left( \text{head} \right) = - \sum_{i \in I}^{}{\sum_{j \in J}^{}{\alpha_{i,j}\left( \mu_{i} - j \right)^{2}}}\ \ \ \ \ \ \ \ \mu_{i} = \sum_{j \in J}^{}{\text{j.}\alpha_{i,j}}$$

-   **Coverage:**\
    It measures the amount of attention a source token has received.

$$\text{cov}\left( \text{head} \right) = \sum_{j \in J}^{}\left( \sum_{i \in I}^{}\alpha_{i,j} \right)^{2}$$

According to the paper, the most important heads are
language-independent as you can see in the following figure:

<div align="center">
    <img src="media/multilinguality_in_transformers/image1.png" width=450>
</div>

> **Note:**\
Even though most important heads are language-independent, in the
paper they showed that it is possible to find the rare heads specific to
a language pair via the extensive SBS (sequential backward selection)
procedure.
