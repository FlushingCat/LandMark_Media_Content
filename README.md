# 🏙️LandMark

<p align="center">
    <picture>
    <img src="https://github.com/FlushingCat/LandMark_media_content/blob/main/Icon.png?raw=true" width="750">
    </picture>
</p>

<p align="center"> <font size="4"> 🌏NeRF the globe if you want </font> </p>

<p align="center">
<video controls width="500">
<source src="https://static.openxlab.org.cn/landmarks/pc/%E5%A4%A7%E5%9C%BA%E6%99%AF%E5%BB%BA%E6%A8%A1_1K_0711.mp4" type="video/mp4">
</video>
<video controls width="500">
<source src="https://static.openxlab.org.cn/landmarks/pc/%E6%99%BA%E5%A1%94%2B%E4%B8%AD%E5%9B%BD%E9%A6%86%2B%E6%AD%A6%E5%BA%B7%E5%A4%A7%E6%A5%BC_1K_0711.mp4" type="video/mp4">
</video>
</p>

<p align="center">
    <a href="https://landmark.intern-ai.org.cn/">
    <font size="4">
    🏠HomePage
    </font>
    </a>
    <a href="https://city-super.github.io/gridnerf/">
    <font size="4">
    ✍️PaperPage
    </font>
    </a>
</p>

# 💻 About
This repository contains the open source code for the Landmark project. The first 3D real-scene large-model method to reconstruct scenes under city scale without limit.<br>
The backbone of the project is the Grid-NeRF and further extended by the dedicated  framework, algorithm and kernels improvements.<br>
Your likes and contributions are exactly what we need. Please Refer to the paper for more detail.
# 🚀 Quickstart
## Prerequisites
You must have a NVIDIA video card with CUDA installed on the system. This library has been tested with version `11.3` of CUDA.
## Install LandMark
The LanMark repository contains configuration files to help you create a proper environment
```
git clone `the repository link`
```
## Create environment
We recommend using `Conda` and the given `environmet.yaml` to manage complicated dependencies:
```
cd LandMark
conda env create -f environment.yaml
```
Don't forget to change the prefix to the directory you expect
## Prepare dataset
Large scale scenes captured from the real world are most suitable for our method.<br>
We recommand using dataset of a building, a well-known landmark and even a small town.<br>
reform your dataset as the following structure:
```
- your_dataset
    |- images
        |- image_0.png
        |- image_1.png
        |- image_2.png
        |- ...
    |-transform_test.json
    |-transform_train.json
```
the `transform` files can be extracted and genderated by `COLMAP` 
## Set arguments
We provide a configuration file `confs/config_example.txt` as an example to help you initialize your experiments.<br>
There are a bunch of arguments for customization. We divide them into  four types for better understanding<br> 
Some important arguments are demonstrated here. don't forget to specify path-related arguments before starting.<br>
```
-experiment   
    |-datadir   Path of your dataset. It's recommended to put it under the LandMark/datasets
    |-dataset_name  Set the type of dataloader rather than the dataset. Using "zhita" as recommended 
    |-basedir   Where to save your training checkpoint. Using LandMark/log by default
-train
    |-start_iters   Number of start iteration in training
    |-n_iters   Total number of iterations in training
    |-batch_size    Training batch size
-render
    |sampling_opt   Whether to use sampling optimization when rendering
```
For more details about arguments, refering to the `LandMark/app/config_parser.py`
## Train your model
Now it's time to train your own GridNeRF model:
```
cd LandMark
python app/trainer.py --config confs/config_example.txt 
```
The training checkpoints and images will be saved in `LandMark/log/your_expname` by default
## Render some images
After the traning process completed, independent rendering test is available:
```
cd LandMark
python app/trainer.py --config confs/zhita.txt --ckpt=log/your_expname/your_expname.th
```
The rendering result will be save in `LandMark/log/your_expname/imgs_test_all` by default
# 📖Learn More
## Directory structure
- app 
    - models    contains sequential and parallel implementations of Grid-NeRF
    - tools     contains dataloaders, train/render utilities
    - trainer.py    manage running process for training
    - renderer.py   manage running process for training
- confs     contains confiuration files for exmeriments
- framework     reserve for future work
- environment.yaml environment confiuration file for conda
- requirements.txt environment confiuration file for pip
## Torch DDP support
TODO
## Parallel Methods
TODO
# 🎨 Support Features
TODO:support features here
# 🤝 Authors
TODO:cluster of human names here
# 📄 License
TODO:support license here