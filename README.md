# MADA-Attack
Official repository of the ICML'26 paper "MADA-Attack: Transferable Multi-modal Attention Distraction Adversarial Attack against Vision Language Models"

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10.8-blue?style=flat&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c?style=flat&logo=pytorch&logoColor=white">
  <a href="https://huggingface.co/AntlersTheWarden/MADA">
    <img src="https://img.shields.io/badge/🤗%20HuggingFace-MADA-yellow?style=flat">
  </a>
  <img src="https://img.shields.io/badge/MIT%20License-green?style=flat">
</p>

## TODO
- [ ] Release training codes
- [x] Release checkpoint and inference sample

## Getting Started
### Inference
We provide UAP checkpoint on [Hugging Face](https://huggingface.co/AntlersTheWarden/MADA/tree/main), and you can follow the instructions in the HF repo or use [Inference Sample](inference.py) with the script below.
```
python inference.py --images dataset_path/ --output_dir output_path/
```
