# One or Two Things We know about Concept Drift -- A Survey on Monitoring Evolving Environments

Experimental code of survey paper. [Part A](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1330257/pdf) [Part B](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1330258/pdf)

## Abstract

The world surrounding us is subject to constant change. These changes, frequently described as concept drift, influence many industrial and technical processes. As they can lead to malfunctions and other anomalous behavior, which may be safety-critical in many scenarios, detecting and analyzing concept drift is crucial.
In this paper, we provide a literature review focusing on concept drift in unsupervised data streams. While many surveys focus on supervised data streams, so far, there is no work reviewing the unsupervised setting. However, this setting is of particular relevance for monitoring and anomaly detection which are directly applicable to many tasks and challenges in engineering.

This survey provides a taxonomy of existing work on drift detection. Besides, it covers the current state of research on drift localization in a systematic way.  In addition to providing a systematic literature review, this work provides precise mathematical definitions of the considered problems and contains standardized experiments on parametric artificial datasets allowing for a direct comparison of different strategies for detection and localization. Thereby, the suitability of different schemes can be analyzed systematically and guidelines for their usage in real-world scenarios can be provided. Finally, there is a section on the emerging topic of explaining concept drift.

**Keywords:** concept drift, drift detection, drift localization, drift explanation, monitoring, explainability, survey 

## Requirements

* Python 
* Numpy, SciPy, Pandas, Matplotlib
* scikit-learn
* chpt (Kernel Drift Detector)

## Cite

Cite our Papers:

Part A
```
@article{hinder2024one,
  title={One or two things we know about concept drift—a survey on monitoring in evolving environments. Part A: detecting concept drift},
  author={Hinder, Fabian and Vaquet, Valerie and Hammer, Barbara},
  journal={Frontiers in Artificial Intelligence},
  volume={7},
  pages={1330257},
  year={2024},
  publisher={Frontiers Media SA}
}
```

Part B
```
@article{hinder2024one,
  title={One or two things we know about concept drift—a survey on monitoring in evolving environments. Part B: locating and explaining concept drift},
  author={Hinder, Fabian and Vaquet, Valerie and Hammer, Barbara},
  journal={Frontiers in Artificial Intelligence},
  volume={7},
  pages={1330258},
  year={2024},
  publisher={Frontiers Media SA}
}
```

[ArXiv version](https://arxiv.org/abs/2310.15826)
```
@misc{hinder2023things,
      title={One or Two Things We know about Concept Drift -- A Survey on Monitoring Evolving Environments}, 
      author={Fabian Hinder and Valerie Vaquet and Barbara Hammer},
      year={2023},
      eprint={2310.15826},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

This code has a MIT license.
