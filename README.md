# Backtranslator Class for Meta FairSeq2 Models

This is a backtranslation library for Meta's [fairseq2](https://github.com/facebookresearch/fairseq2) library. Backtranslation is a data augmentation technique for low-resource unsupervised machine translation. When parallel data pairs are insufficient for effective translation, backtranslation leverages more abundant monolingual data[1][2].

Given monolingual data for a source language $$X$$, backtranslation augments parallel data pairs by translating a sentence $$x \in X$$ to a sentence $$y \in Y$$ in intermediate language $$Y$$. Reverse-translation of $$y$$ back into $$X$$ should produce a sentence $$\tilde{x}$$ such that $$\tilde{x} \approx x$$. The model is then updated using cross-entropy loss to minimize the difference between $$x$$ and $$\tilde{x}$$, effectively improving translation in both directions[7].

## Read more about backtranslation

For more information on unsupervised neural machine translation and backtranslation techniques, refer to the following papers:

1. [Artetxe, M., Labaka, G., Agirre, E., & Cho, K. (2018). Unsupervised Neural Machine Translation. In International Conference on Learning Representations](https://arxiv.org/abs/1710.11041).

2. [Lample, G., Conneau, A., Denoyer, L., & Ranzato, M. (2019). Unsupervised Machine Translation Using Monolingual Corpora Only. In International Conference on Learning Representations](https://arxiv.org/abs/1711.00043).

# This Project

This is a personal project aimed at better understanding the fairseq2 library, specifically by fine-tuning the SONAR text-to-text model. This repository contains a `backtranslator.py` class with a generic backtranslator module that can extend to any of Fairseq2's models.
