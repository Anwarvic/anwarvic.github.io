---
title: The Effect of Alignment Objectives on Code-Switching Translation
date: 2022-06-30
authors: ["Mohamed Anwar"]
labs: ["African Institute for Mathematical Sciences (AIMS)"]
link: https://arxiv.org/pdf/2309.05044.pdf
repo:
poster:
demo:
slides:
talk:
comments: false
layout: my_research_post
---

One of the things that need to change when it comes to machine translation is
the models' ability to translate code-switching content, especially with the
rise of social media and user-generated content. In this paper, we are
proposing a way of training a single machine translation model that is able to
translate monolingual sentences from one language to another, along with
translating code-switched sentences to either language. This model can be
considered a bilingual model in the human sense. For better use of parallel
data, we generated synthetic code-switched (CSW) data along with an alignment
loss on the encoder to align representations across languages. Using the WMT14
English-French (En-Fr) dataset, the trained model strongly outperforms
bidirectional baselines on code-switched translation while maintaining quality
for non-code-switched (monolingual) data.


## Paper Citation

{%
    include layeredImage.html
    cover="/paper.png"
    conference="arXiv (2309.05044)"
    bibtex="https://scholar.googleusercontent.com/scholar.bib?q=info:OWPpKPurTVAJ:scholar.google.com/&output=citation&scisdr=ClEsqv87EPzihSojdpQ:AFWwaeYAAAAAZYIlbpTXvPKrJQuCVkEPNcyIM4w&scisig=AFWwaeYAAAAAZYIlbpY-W_ertVdeK-FV0k7WJO8&scisf=4&ct=citation&cd=-1&hl=en"
%}
