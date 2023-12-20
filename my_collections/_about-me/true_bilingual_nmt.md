---
cover: /cover.png
title: True Bilingual Neural Machine Translation
date: 2022-04-08
authors: ["Mohamed Anwar", "Lekan Raheem", "Maab Elrashid", "Melvin Johnson", "Julia Kreutzer"]
labs: ["African Institute for Mathematical Sciences (AIMS)", "Google Research"]
link: https://openreview.net/pdf?id=SAGNK9ME8Wc
repo: 
poster: https://drive.google.com/file/d/1TSeM-UTHqtl4dYlOkpd_5nFN3owP6Ga-/view?usp=sharing
demo: https://github.com/Anwarvic/truel_nmt
slides: https://docs.google.com/presentation/d/1dS-kLyHegtuoONahjx4YKQWFx7B_w3rfS2wVOrq0K6U/edit?usp=sharing
talk:
comments: false
layout: my_research_post
---

Bilingual machine translation permits training a single model that translates
monolingual sentences from one language to another. However, a model is not
truly bilingual unless it can translate back and forth in both language
directions it was trained on, along with translating code-switched sentences to
either language. We propose a true bilingual model trained on WMT14
English-French (En-Fr) dataset. For better use of parallel data, we generated
synthetic code-switched (CSW) data along with an alignment loss on the encoder
to align representations across languages. Our model strongly outperforms
bilingual baselines on CSW translation while maintaining quality for non-code
switched data. 


## Methodology

Generate realistic code-switching translation using the following method:

<div align="center">
    <a href="https://github.com/facebookresearch/muavic?tab=readme-ov-file#getting-data">
        <img src="media/true_bilingual_nmt/csw_data.png" width=750>
    </a>
</div>

Use an alignment objective on the encoder side to create language-agnostic representations:

<div align="center">
    <a href="https://github.com/facebookresearch/muavic?tab=readme-ov-file#getting-data">
        <img src="media/true_bilingual_nmt/alignment_method.png" width=750>
    </a>
</div>


## Paper Citation

{%
    include layeredImage.html
    cover="/paper.png"
    conference="ICLR 2022 Workshop AfricaNLP"
    bibtex="https://scholar.googleusercontent.com/scholar.bib?q=info:IjfV-mb6r9EJ:scholar.google.com/&output=citation&scisdr=ClEsqv87EPzihSoifvA:AFWwaeYAAAAAZYIkZvBoiZnQDmMk9KlJGgVi7s8&scisig=AFWwaeYAAAAAZYIkZl9eVdy7aT5QIBppt6_jjiM&scisf=4&ct=citation&cd=-1&hl=en"
%}
