# RMA Task Enhancements: HammeringRMA & ScrewdriverRMA

![MIT License](https://img.shields.io/github/license/JohnsonOfori/RMA-Task-Enhancements)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Repo Size](https://img.shields.io/github/repo-size/JohnsonOfori/RMA-Task-Enhancements)
![Last Commit](https://img.shields.io/github/last-commit/JohnsonOfori/RMA-Task-Enhancements)
![Issues](https://img.shields.io/github/issues/JohnsonOfori/RMA-Task-Enhancements)

## 📘 Project Overview

This repository provides simulation environments and task-specific code developed as part of an MPhil project at KNUST to enhance the CVPR 2024 paper *"Rapid Motor Adaptation for Robotic Manipulator Arms"* by Liang, Ellis, and Henriques.

The environments extend the original work by introducing fine-motor manipulation tasks such as:

- 🛠️ **HammeringRMA** – for contact-intensive impact control
- 🔩 **ScrewdriverRMA** – for torque-driven rotational actions

## 📂 Repository Contents

- `hammer.py`: Hammering simulation logic for robotic arm adaptation
- `screw.py`: Screw-driving environment definition
- `requirements.txt`: Dependency file for setting up the environment
- `LICENSE`: MIT license for reuse and academic citation

## 🚀 Installation

```bash
git clone https://github.com/JohnsonOfori/RMA-Task-Enhancements.git
cd RMA-Task-Enhancements
pip install -r requirements.txt
```

Make sure you have the [ManiSkill2](https://github.com/haosulab/ManiSkill2) framework and SAPIEN installed and configured properly for simulation.

## ▶️ Usage

You can run each simulation environment with:

```bash
python hammer.py
python screw.py
```

Customize parameters inside each file or integrate with a higher-level RMA training pipeline.

## 📜 Citation

If you use this work in your research, please cite:

```
@misc{johnson2025rma,
  author = {Johnson Ofori Amanfo},
  title = {Enhanced Rapid Motor Adaptation for Robotic Manipulator Arms},
  note = {MPhil Project, KNUST, Ghana. Extension of CVPR 2024 work by Liang et al.},
  year = {2025}
}
```

## 🔗 Related Paper

Original paper available at: [CVPR arXiv preprint](https://arxiv.org/abs/2312.04670)

---

Developed and maintained by [Johnson Ofori Amanfo](https://github.com/JohnsonOfori)
