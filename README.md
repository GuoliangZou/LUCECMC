# Learning Dual Enhanced Representation for Contrastive Multi-view Clustering

Training implementation of **Learning Dual Enhanced Representation for Contrastive Multi-view Clustering (LUCE-CMC)**. The model class in this repository is named `LUCECMC`.

> **Paper:** Guoliang Zou, Yangdong Ye, Tongji Chen, Shizhe Hu, *Learning Dual Enhanced Representation for Contrastive Multi-view Clustering*, **Proceedings of the 32nd ACM International Conference on Multimedia (MM '24)**, pp. 8731-8739, 2024.  
> **DOI:** [10.1145/3664647.3681030](https://doi.org/10.1145/3664647.3681030)

## Overview

LUCE-CMC learns dual enhanced representations through two complementary mechanisms:

- **Enhanced feature-level contrastive learning (En-FeaCL):** combines cross-view feature alignment with reconstruction (called reconstitution in the paper) to reduce the influence of noise and learn discriminative representations.
- **Enhanced cluster-level contrastive learning (En-CluCL):** contrasts each view's cluster assignments with those obtained from shared fused features, introducing global information into cluster-level learning.

A **deep divergence-based clustering (DDC)** module produces the final clustering assignments. The paper combines DDC, reconstruction, feature alignment, and enhanced cluster-level contrastive losses in its overall objective (Eq. 9).



## Environment

The supplied Conda environment specifies Python 3.8. Run from the repository root:

```powershell
conda env create -f environment.yml
conda activate mvc
python -m pip install -r requirements.txt
python -m pip install 'pydantic<2' matplotlib seaborn
```

Configuration classes require Pydantic 1.x. The current loss implementation calls `.cuda()` directly, so training requires a CUDA-capable NVIDIA GPU and compatible PyTorch installation. Dependency versions are not fully pinned; these commands have not been verified with a complete training run.

## Training

Run from the repository root. In PowerShell, add `src` to `PYTHONPATH` so that the training packages resolve correctly:

```powershell
$env:PYTHONPATH = "$($PWD.Path)\src"
python -m models.train -c caltech3v
```

Alternatively, run `python -m models.train -c caltech3v` from the `src` directory.

Defaults are **20 runs**, up to **100 epochs** per run, batch size **100**, and Adam with learning rate **0.001**. Results are saved to `models/caltech3v-<experiment-id>/run-<run-index>/`. The best checkpoint, `best.pt`, is selected by the lowest evaluation `ddc_1` loss.

For a short initial check that reaches the default evaluation interval:

```powershell
python -m models.train -c caltech3v --n_runs 1 --n_epochs 4
```

### Evaluation

Using the same environment and `PYTHONPATH`, evaluate saved runs:

```powershell
python -m models.evaluate -c caltech3v -t <experiment-id>
```

Replace `<experiment-id>` with the suffix of the `models/caltech3v-...` directory. The included experiment uses `-t 22pxhgp2`.



## Citation

If this code is useful in your research, please cite:

```bibtex
@inproceedings{zou2024lucecmc,
  title     = {Learning Dual Enhanced Representation for Contrastive Multi-view Clustering},
  author    = {Zou, Guoliang and Ye, Yangdong and Chen, Tongji and Hu, Shizhe},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  series    = {MM '24},
  pages     = {8731--8739},
  year      = {2024},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3664647.3681030}
}
```

The base implementation is attributed to SiMVC/CoMVC and *Reconsidering Representation Alignment for Multi-view Clustering* (CVPR 2021), cited as reference [22] in the LUCE-CMC paper.

## Publication

- Conference: **32nd ACM International Conference on Multimedia (MM '24)**
- Location: **Melbourne, VIC, Australia**
- Dates: **October 28-November 1, 2024**
- Pages: **8731-8739**
- DOI: [10.1145/3664647.3681030](https://doi.org/10.1145/3664647.3681030)

## License

The repository's [LICENSE](LICENSE) is the MIT License, with copyright attributed to DanielTrosten (2021). Transformer source files also contain Clear BSD notices from Shanghai Yitu Technology; retain the corresponding source notices.


