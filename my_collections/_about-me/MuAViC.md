---
cover: /cover.png
title: "MuAViC: A Multilingual Audio-Visual Corpus for Robust Speech Recognition and Robust Speech-to-Text Translation"
date: 2023-01-01
authors: ["Mohamed Anwar", "Bowen Shi", "Vedanuj Goswami", "Wei-Ning Hsu", "Juan Pino", "Changhan Wang"]
labs: ["Meta AI"]
link: https://www.isca-speech.org/archive/pdfs/interspeech_2023/anwar23_interspeech.pdf
repo: https://github.com/facebookresearch/muavic
poster:
demo: https://github.com/facebookresearch/muavic/tree/main/demo
slides: https://docs.google.com/presentation/d/1w3OnsWXmAlqDlrwN-kUJsp1wrLn17kBy8eOKzv9aMsA/edit?usp=sharing
talk:
comments: false
layout: my_research_post
---

We introduce MuAViC, a multilingual audio-visual corpus for robust speech
recognition and robust speech-to-text translation providing 1200 hours of
audio-visual speech in 9 languages. It is fully transcribed and covers 6
English-to-X translation as well as 6 X-to-English translation directions. To
the best of our knowledge, this is the first open benchmark for audio-visual
speech-to-text translation and the largest open benchmark for multilingual
audio-visual speech recognition. Our baseline results show that MuAViC is
effective for building noise-robust speech recognition and translation models.
We make the corpus available at
[https://github.com/facebookresearch/muavic](https://github.com/facebookresearch/muavic).

<div align="center">
    <img src="media/MuAViC/teaser.png" width=750>
</div>

## Download Dataset

<div align="center">
    <a href="https://github.com/facebookresearch/muavic?tab=readme-ov-file#getting-data">
        <img src="media/MuAViC/dataset.jpg" width=750>
    </a>
</div>


## Demo

Try out our [interactive demo](https://github.com/facebookresearch/muavic/tree/main/demo):

{% include twitterPlayer.html id="anwarvic_/status/1690481769076719616" %}

## Methodology

To create en->x direction in MuAViC, we combined LRS3-TED dataset with TED2020 dataset
by matching the English transcripts in both:

<div align="center">
    <a href="https://github.com/facebookresearch/muavic?tab=readme-ov-file#getting-data">
        <img src="media/MuAViC/en_x.gif" width=750>
    </a>
</div>

To create x->en direction in MuAViC, we downloaded the TED talks from YouTube
based on the mTEDx talk-id:

<div align="center">
    <a href="https://github.com/facebookresearch/muavic?tab=readme-ov-file#getting-data">
        <img src="media/MuAViC/x_en.gif" width=750>
    </a>
</div>


## Paper Citation

{%
    include layeredImage.html
    cover="/paper.png"
    conference="INTERSPEECH 2023"
    bibtex="https://scholar.googleusercontent.com/scholar.bib?q=info:3ezj-DWwmCEJ:scholar.google.com/&output=citation&scisdr=ClF1R_RmEPzihSnQMeo:AFWwaeYAAAAAZYHWKerW8ip7nqB-osKWeeEvzxU&scisig=AFWwaeYAAAAAZYHWKePxDkC7BnXfujqJMC0JJ20&scisf=4&ct=citation&cd=-1&hl=en"
%}
