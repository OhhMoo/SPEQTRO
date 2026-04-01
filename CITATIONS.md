# Citations and Attributions

This project integrates several external tools and models developed by academic research groups. Please cite the original authors when using these components.

## Integrated Tools

### CASCADE 2.0
**¹³C NMR Chemical Shift Prediction**
- **Authors:** Abhijeet Bhadauria, Zhitao Feng, Mihai Popescu, Robert Paton — Paton Lab, Colorado State University
- **Paper:** Bhadauria et al., "CASCADE-2.0: Real Time Prediction of ¹³C-NMR Shifts with sub-ppm Accuracy" (2025)
- **DOI:** https://doi.org/10.26434/chemrxiv-2025-r8m9m
- **Repository:** https://github.com/patonlab/CASCADE
- **License:** MIT — Copyright (c) 2025 Abhijeet Bhadauria & Robert Paton / Paton Lab
- **Citation:**
  ```bibtex
  @article{bhadauria2025cascade,
    title={CASCADE-2.0: Real Time Prediction of 13C-NMR Shifts with sub-ppm Accuracy},
    author={Bhadauria, Abhijeet and Feng, Zhitao and Popescu, Mihai and Paton, Robert},
    journal={ChemRxiv},
    year={2025},
    doi={10.26434/chemrxiv-2025-r8m9m}
  }
  ```

### ChefNMR
**NMR-to-Structure Elucidation via Atomic Diffusion Models**
- **Authors:** Ziyu Xiong, Yichi Zhang, Foyez Alauddin, Chu Xin Cheng, Joon Soo An, Mohammad R. Seyedsayamdost, Ellen D. Zhong — Princeton University / MIT
- **Paper:** Xiong et al., "Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra" (NeurIPS 2025)
- **ArXiv:** https://arxiv.org/abs/2512.03127
- **Checkpoints:** https://zenodo.org/records/17766755
- **Repository:** https://github.com/ml-struct-bio/chefnmr
- **License:** MIT — Copyright (c) 2025 Ziyu Xiong
- **Citation:**
  ```bibtex
  @inproceedings{xiongatomic,
    title={Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra},
    author={Xiong, Ziyu and Zhang, Yichi and Alauddin, Foyez and Cheng, Chu Xin and An, Joon Soo and Seyedsayamdost, Mohammad R and Zhong, Ellen D},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    url={https://arxiv.org/abs/2512.03127},
    year={2025}
  }
  ```

### DP5 / DP4
**Bayesian NMR Structure Verification**
- **Authors:** Kristaps Ermanis, Jonathan M. Goodman — Goodman Group, University of Cambridge
- **Paper:** Ermanis et al., "Assigning Stereochemistry to Single Diastereoisomers by GIAO NMR Calculation: The DP4 Probability" (JACS 2010); Smith & Goodman, "Assigning Stereochemistry to Single Diastereoisomers by GIAO NMR Calculation: The DP4 Probability" (JACS 2010)
- **Repository:** https://github.com/Goodman-lab/DP5
- **License:** MIT — Copyright (c) 2015 Kristaps Ermanis, Jonathan M. Goodman; Copyright (c) 2022 Goodman Lab
- **Citation:**
  ```bibtex
  @article{dp4_2010,
    title={Assigning Stereochemistry to Single Diastereoisomers by GIAO NMR Calculation: The DP4 Probability},
    author={Smith, Steven G. and Goodman, Jonathan M.},
    journal={Journal of the American Chemical Society},
    volume={132},
    number={37},
    pages={12946--12959},
    year={2010}
  }
  ```

### SSIN
**Substructure-Directed Spectrum Interpreter Network for IR Spectroscopy**
- **Authors:** Authors listed on the ACS publication (GitHub: ngs00)
- **Paper:** "Explainable Machine Learning for Characterizing Unknown Molecular Structures in Infrared Spectra" (Analytical Chemistry 2025)
- **DOI:** https://doi.org/10.1021/acs.analchem.5c03126
- **Repository:** https://github.com/ngs00/SSIN
- **License:** No license file provided in the repository. **Usage may be restricted — contact authors for permission before redistribution.**
- **Citation:**
  ```bibtex
  @article{ssin2025,
    title={Explainable Machine Learning for Characterizing Unknown Molecular Structures in Infrared Spectra},
    author={{See published article for full author list}},
    journal={Analytical Chemistry},
    volume={97},
    number={38},
    pages={20869--20878},
    year={2025},
    doi={10.1021/acs.analchem.5c03126}
  }
  ```

### ms-pred / ICEBERG
**Mass Spectrometry Fragmentation Prediction**
- **Authors:** Samuel Goldman, Janet Li, Runzhong Wang, Mrunali Manjrekar, Connor W. Coley — MIT CSAIL / Coley Group
- **Papers:**
  - Goldman et al., "Generating Molecular Fragmentation Graphs with Autoregressive Neural Networks" (Analytical Chemistry 2024)
  - Wang et al., "Neural Spectral Prediction for Structure Elucidation with Tandem Mass Spectrometry" (bioRxiv 2025)
- **Repository:** https://github.com/samgoldman97/ms-pred
- **License:** MIT — Copyright (c) 2023 Samuel Goldman
- **Citations:**
  ```bibtex
  @article{goldman2024generating,
    title={Generating molecular fragmentation graphs with autoregressive neural networks},
    author={Goldman, Samuel and Li, Janet and Coley, Connor W},
    journal={Analytical Chemistry},
    volume={96},
    number={8},
    pages={3419--3428},
    year={2024},
    publisher={ACS Publications}
  }

  @article{wang2025neuralspec,
    author={Wang, Runzhong and Manjrekar, Mrunali and Mahjour, Babak and Avila-Pacheco, Julian and Provenzano, Joules and Reynolds, Erin and Lederbauer, Magdalena and Mashin, Eivgeni and Goldman, Samuel L. and Wang, Mingxun and Weng, Jing-Ke and Plata, Desir{\'e}e L. and Clish, Clary B. and Coley, Connor W.},
    title={Neural Spectral Prediction for Structure Elucidation with Tandem Mass Spectrometry},
    journal={bioRxiv},
    year={2025},
    doi={10.1101/2025.05.28.656653}
  }
  ```

## Python Library Dependencies

This project also relies on the following open-source libraries. These are installed as dependencies and not bundled with speqtro:

| Library | License | Use in speqtro |
|---------|---------|----------------|
| [RDKit](https://www.rdkit.org/) | BSD-3-Clause | Molecular parsing, SMARTS matching, conformer generation |
| [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) | MIT | Claude API client |
| [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) | MIT | Agentic reasoning loop |
| [Typer](https://typer.tiangolo.com/) | MIT | CLI framework |
| [Rich](https://github.com/Textualize/rich) | MIT | Terminal UI and markdown rendering |
| [NumPy](https://numpy.org/) | BSD-3-Clause | Numeric computing |
| [SciPy](https://scipy.org/) | BSD-3-Clause | Scientific functions |
| [Pandas](https://pandas.pydata.org/) | BSD-3-Clause | Data manipulation |
| [Matplotlib](https://matplotlib.org/) | PSF-based | Plotting |

## Usage Guidelines

When publishing research using speqtro, please:
1. Cite the original tool papers for any methods you use
2. Acknowledge speqtro as the integration framework
3. Follow the individual licenses of each integrated tool
4. Note that SSIN has no explicit license — contact the authors before redistribution

## Contributing

If you integrate additional external tools, please update this file with proper attribution.
