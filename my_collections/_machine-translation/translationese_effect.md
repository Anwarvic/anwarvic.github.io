---
title: "Translationese Effect"
date: 2018-08-21
labs: ["University of Zurich", "University of Edinburgh"]
---

Translationese is a common term that refers to to translated texts. The
fundamental law of translation states that "phenomena pertaining to the
make-up of the source text tend to be transferred to the target text"
which means that <u><strong>translated texts tend to be simpler and retain some
characteristics from the source language</strong></u>.

A 2009-published paper: [Automatic detection of translated text and its
impact on machine translation](https://arxiv.org/pdf/1808.07048.pdf) has
found out that an MT system performs better when trained on parallel
data whose source side is original and whose target side is
translationese than if it was trained on the opposite. In other words,
if there is a dataset published that contains English→French
translations, it will be better if we use this dataset that direction
and not the opposite (French→English).

And this 2018-published paper: [The Effect of Translationese in Machine
Translation Test Sets](https://arxiv.org/pdf/1906.08069.pdf) found out
that the effect of translationese tends to be high with low-resource
language which could inflate the expectations in terms of translation
quality for these languages. You can even reproduce their results by
taking a look at their official GitHub repository:
[translationese](https://github.com/jjzha/translationese).
