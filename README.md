# Char-CNN-RNN for PyTorch
This repository contains a PyTorch-compatible version of the `Char-CNN-RNN`
model, as described in the paper ["Learning Deep Representations of
Fine-Grained Visual Descriptions"](https://github.com/reedscot/cvpr2016).

The `Char-CNN-RNN` model is prevalent in the Text-to-Image task, and is used
to process image descriptions to obtain embeddings that contain
visual-relevant features.
This PyTorch translation may be useful for researchers interested in using
`Char-CNN-RNN` models without relying on precomputed embeddings, which is
especially handy for testing models with novel image descriptions.

We provide PyTorch-compatible model weights for two `Char-CNN-RNN` model
instances:
* [Learning Deep Representations of Fine-Grained Visual
  Descriptions](https://github.com/reedscot/cvpr2016)
* [Generative Adversarial Text-to-Image
  Synthesis](https://github.com/reedscot/icml2016)

The supported datasets are:
* [Caltech-UCSD
  Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [Oxford 102 Category
  Flower](http://www.robots.ox.ac.uk/~vgg/data/flowers/102)



## Usage
1. Install repository requirements via `pip install -r requirements.txt`
2. Download the pretrained Char-CNN-RNN models from the original authors:
    * [reed_cvpr2016](https://github.com/reedscot/cvpr2016)
    * [reed_icml2016](https://github.com/reedscot/icml2016)
3. Check `example.py` for instructions on how to extract model weights and how
   to use the provided implementation.



## TODOs
* Reproduce training procedure to train new models and support different datasets
