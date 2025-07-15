# Forbes (ECCV2024)

Jintae Kim,
Seungwon Yang,
Seong-Gyun Jeong,
and Chang-Su Kim

Official code for **"Forbes: Face Obfuscation Rendering via Backpropagation Refinement Scheme"**[[paper]](https://arxiv.org/abs/2407.14170)

### Requirements
- PyTorch 1.13.1
- CUDA 11.6
- python 3.8

### Installation
Download repository:
```bash
$ git clone https://github.com/mcljtkim/Forbes.git
```

Create conda environment:
```bash
$ cd env
$ sh create_env.sh
```

Download AdaFace pre-trained model parameters from (https://github.com/mk-minchul/AdaFace).

Direct link to the parameters: [pre-trained model](https://drive.google.com/file/d/1m757p4-tUU5xlSHLaO04sqnhvqankimN/view).

Generate two folders and put the weights to the "weights" folder.
```bash
$ mkdir output
$ mkdir weights
```

### Quick Usage
Generate an output image
```bash
$ python demo.py --img_path image_path/input_image.png
```

### Evaluation
If you want to evaluate benchmark datasets, please refer to the eval.py file
```bash
$ python eval.py dataset_root $/datasetroot
```
You can download the dataset from [Data Zoo](https://github.com/ZhaoJ9014/face.evoLVe).


### Citation
Please cite the following paper if you feel this repository useful.
```bibtex
@inproceedings{kim2024Forbes,
    author      = {Kim, Jintae and Yang, Seungwon and Jeong, Seong-Gyun and Kim, Chang-Su},
    title       = {Forbes: Face Obfuscation Rendering via Backpropagation Refinement Scheme},
    booktitle   = {Eur. Conf. Comput. Vis.},
    year        = {2024}
}
```
