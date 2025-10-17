# DEARLi

**Official implementation of the ICCV 2025 Findings paper:**  
**"DEARLi: Decoupled Enhancement of Recognition and Localization for Semi-Supervised Panoptic Segmentation"**

---

## Getting Started

For detailed setup instructions -- including installation, dataset preparation, checkpoints, training, and evaluation -- please see:  
[`readmes/GETTING_STARTED.md`](./readmes/GETTING_STARTED.md)

---

## TODO

- [ ] Upload COCO-Objects dataset generation script and panoptic labels  
- [ ] Upload SAM-generated pseudolabels and add instructions to generate them  

---

## Acknowledgments

This project builds upon several excellent open-source frameworks.  
We thank the respective authors for making their code publicly available:

- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [FC-CLIP](https://github.com/bytedance/fc-clip)

---

## Citation

If you use **DEARLi** in your research, please cite the following:

```bibtex
@article{martinovic2025dearli,
  title={DEARLi: Decoupled Enhancement of Recognition and Localization for Semi-Supervised Panoptic Segmentation},
  author={Martinović, Ivan and Šarić, Josip and Oršić, Marin and Kristan, Matej and Šegvić, Siniša},
  journal={arXiv preprint arXiv:2507.10118},
  year={2025}
}