# Softmax is not Enough (for Adaptive Conformal Classification)
**Official code for the paper presented at [ICLR 2026](https://iclr.cc/).**

**Authors:** Navid Akhavan Attar, Hesam Asadollahzadeh, Ling Luo, Uwe Aickelin 

- **OpenReview:** [ICLR 2026 submission](https://openreview.net/forum?id=zCwTMRtASZ)

## Code

This codebase builds on [TorchCP](https://github.com/ml-stat-Sustech/TorchCP), a PyTorch toolbox for conformal prediction.

We retain the original TorchCP library code and add our implementation for Energy-Based Conformal Classification. The main additions in this repository are:

- examples/classification_energy_cp.py
- EnergyAPS in torchcp/classification/score/aps.py
- EnergyRAPS in torchcp/classification/score/raps.py
- EnergySAPS in torchcp/classification/score/saps.py
- EnergyLAC in torchcp/classification/score/lac.py

## Citation

If you use this repository, please cite our paper:

```bibtex
@inproceedings{attarsoftmax,
  title={Softmax is not Enough (for Adaptive Conformal Classification)},
  author={Attar, Navid Akhavan and Asadollahzadeh, Hesam and Luo, Ling and Aickelin, Uwe},
  booktitle={The Fourteenth International Conference on Learning Representations}
}