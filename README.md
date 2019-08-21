# Char-CNN-RNN for PyTorch
PyTorch version of the `Char-CNN-RNN` model and train/evaluation procedures,
as described in the paper ["Learning Deep Representations of Fine-Grained
Visual Descriptions"](https://github.com/reedscot/cvpr2016).

This repository also provides instructions on how to extract and use the
original weights for two papers:
* [Learning Deep Representations of Fine-Grained Visual
  Descriptions](https://github.com/reedscot/cvpr2016)
* [Generative Adversarial Text-to-Image
  Synthesis](https://github.com/reedscot/icml2016)


## Requirements
This implementation requires PyTorch >= 1.1.0 and Python 3. Check
`requirements.txt` file for information on other packages.


## Training and Evaluation
The `scripts` folder contains bash scripts that reproduce the original training
and evaluation procedures.

1. Install repository requirements via `pip install -r requirements.txt`
2. Download the datasets from the original author
   [here](https://github.com/reedscot/cvpr2016).
3. Run `python sje_train -h` to get instructions (or check `script` folder).
   You can open TensorBoard to check live training results.
4. After training, run `python sje_eval -h` to get instructions for evaluation
   procedures.


## Using Original Weights
1. Download the pretrained Char-CNN-RNN models from the incarnation you desire:
    * [reed_cvpr2016](https://github.com/reedscot/cvpr2016)
    * [reed_icml2016](https://github.com/reedscot/icml2016)
2. Check `example.py` for instructions on how to extract model weights and how
   to use the provided implementation.

The `Char-CNN-RNN` model is prevalent in the Text-to-Image task, and is used to
process image descriptions to obtain embeddings that contain visual-relevant
features. This PyTorch translation may be useful for researchers interested in
using `Char-CNN-RNN` models without relying on precomputed embeddings, which is
especially handy for testing models with novel image descriptions or new
datasets.


## TODOs
* Add instructions on how to process custom datasets.
