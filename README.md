<p align="center">
    <img src="https://img.shields.io/badge/Trainer-Ready-green"/>
    <img src="https://img.shields.io/badge/Renderer-Ready-green"/>
    <img src="https://img.shields.io/badge/Framework-Developing-blue"/>
    <img src="https://img.shields.io/badge/Documentation-Preview-purple"/>
    <img src="https://img.shields.io/badge/License-MIT-orange"/>
</p>


<p align="center">
    <picture>
    <img src="https://github.com/FlushingCat/LandMark_Media_Content/blob/main/logo.png?raw=true" width="750">
    </picture>
</p>

<p align="center"> <font size="4"> 🌏NeRF the globe if you want </font> </p>

<p align="center">
<picture>
    <img src="https://github.com/FlushingCat/LandMark_Media_Content/blob/main/zhujiajiao.gif?raw=true" width="300">
    </picture>
    <picture>
    <img src="https://github.com/FlushingCat/LandMark_Media_Content/blob/main/wukang_noblackscreen.gif?raw=true" width="300">
    </picture>
    <picture>
    <img src="https://github.com/FlushingCat/LandMark_media_content/blob/main/xian.gif?raw=true" width="300">
    </picture>
    <picture>
    <img src="https://github.com/FlushingCat/LandMark_Media_Content/blob/main/sanjiantao.gif?raw=true" width="300">
    </picture>
</p>

<p align="center">
    <a href="https://landmark.intern-ai.org.cn/">
    <font size="4">
    🏠HomePage
    </font>
    </a>
    |
    <a href="https://flushingcat.github.io/LandMark-Documentation-Site/">
    <font size="4">
    📑DocumentationSite
    </font>
    </a>
    |
    <a href="https://city-super.github.io/gridnerf/">
    <font size="4">
    ✍️PaperPage
    </font>
    </a>
</p>

# 💻 About
This repository contains the open source code for the project LanMark. The first large-model method aiming at reconstructing city-scale scenes.<br>
The backbone of the project is the Grid-NeRF. Please refer to the paper for more detailed information about the elementary algorithm.<br>
Based on Grid-NeRF, the LandMark made enormous number of extentions and overhaul the dedicated  systems, algorithms and operators.<br>
There are many one-of-a-kind highlights in the LandMark:

- Large-scale, high-precision Reconstrution:
    - For the first time, efficient training of 100 square kilometers of city-scale NeRF was realized, and the rendering resolution reached 4K with model parameters over 200 billions.
- Multiple feature extentions:
    - Rich capabilities beyond reconstruction, including ajusting urban layout such as removing or adding a buildings, and ajusting appearance style such as lighting changes related to seasons.
- Training, rendering integrated system:
    - City-scale NeRF system covering algorithms, operators, computing systems, which provides a solid foundation for the training, rendering and application of real-world 3D large models.

And now it's possible to train and render with your own LandMark models and enjoy your creativity.<br>
Your likes and contributions to the community are exactly what we need.
# 🎨 Support Features
The LandMark have supported plenty of features at present:

- Pytorch DDP both on training and rendering
- Sequential Model Traning
- Parrallel Model Traning
    - Branch Parrallel
    - Plane Parrallel
    - Channel Parrallel
- Sequential Model Rendering

It's highly recommanded to read the [DOCUMENTATION](https://flushingcat.github.io/LandMark-Documentation-Site/) about our implementations of the acceleration strategies. 
# 🚀 Quickstart
## Prerequisites
You must have a NVIDIA video card with CUDA installed on the system. This library has been tested with version `11.6` of CUDA.
## Install LandMark
The LanMark repository files contains configuration files to help you create a proper environment
```
git clone `the repository link`
```
## Create environment
We recommend using `Conda` to manage complicated dependencies:
```
conda create --name landmark -y python=3.9.16
conda activate landmark
python -m pip install --upgrade pip
```
This library has been tested with version `3.9.16` of Python.
## Pytorch & CUDA
Install pytorch with CUDA using the commands below once and for all:
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
## Dependencies
We also provide `requirement.txt` for setting environment directly. 
```
cd landmark
pip install -r requirement.txt
```
## Prepare dataset
Large scale scenes captured from the real world are most suitable for our method.<br>
We recommand using dataset of a building, a well-known landmark and even a small town.<br>
Reform your dataset as the following structure:

- your_dataset/
    - images/
        - image_0.png
        - image_1.png
        - image_2.png
        - ...
    - transform_test.json
    - transform_train.json

the poses in the `transform` files can be extracted and genderated by [COLMAP](https://colmap.github.io/) as other NeRF methods.
## Set arguments
We provide a configuration file `confs/config_example.txt` as an example to help you initialize your experiments.<br>
There are bunches of arguments for customization. We divide them into  four types for better understanding<br> 
Some important arguments are demonstrated here. don't forget to specify path-related arguments before procceding.<br>

- experiment   
    - datadir - Path of your dataset. It's recommended to put it under the LandMark/datasets
    - dataset_name - Set the type of dataloader rather than the dataset. Using "zhita" as recommended 
    - basedir - Where to save your training checkpoint. Using LandMark/log by default
- train
    - start_iters - Number of start iteration in training
    - n_iters - Total number of iterations in training
    - batch_size - Training batch size
- render
    - sampling_opt - Whether to use sampling optimization when rendering
- model
    - ... - Hyperparameters to define a LandMark model

For more details about arguments, refer to the `Landmark/app/config_parser.py`
## Train your model
Now it's time to train your own LandMark model:
```
python app/trainer.py --config confs/config_example.txt 
```
The training checkpoints and images will be saved in `Landmark/log/your_expname` by default
## Render some images
After the traning process completed, independent rendering test is available:
```
python app/renderer.py --config confs/config_example.txt --ckpt=log/your_expname/your_expname.th
```
The rendering results will be save in `Landmark/log/your_expname/imgs_test_all` by default
# 📖Learn More
## Directory structure

- app/ 
    - models/ - contains sequential and parallel implementations of models
    - tools/ - contains dataloaders, train/render utilities
    - trainer.py - manage running process for training
    - renderer.py - manage running process for training
- confs/ - contains confiuration files for experiments
- framework/ - reserve for future work
- environment.yaml - environment confiuration file for conda
- requirements.txt - environment confiuration file for pip

## Distributed Data Parrallel support
The trainer and the renderer are both support pytorch DDP originally.<br>
To train with DDP, use command below:
```
python -m torch.distributed.launch --nproc_per_node=number_of_GPUs app/trainer.py --config confs/config_example.txt
``` 
To render with DDP, use command below:
```
python -m torch.distributed.launch --nproc_per_node=number_of_GPUs app/renderer.py --config confs/config_example.txt --ckpt=log/your_expname/your_expname.th
```
Some arguments related to the multi-GPU environment might need to be set properly. Specify `number_of_GPUs` according to your actual environment.
## Train with Parrallel Methods
There three types of parrallel strategies are currently supported for training.<br>
To involve these parralel features in your experiments, simply use the configuration files such as `confs/branch_parrallel_config_example.txt`.<br>
After changing the path arguments in the configuration file, you are ready to train a plug-and-play branch parrallel model:
```
python -m torch.distributed.launch --nproc_per_node=number_of_GPUs app/trainer.py --config confs/branch_parrallel_config_example.txt
```
There are few differences in use between training a branch parralel model and a sequential model with DDP, but the training effeciency meet great acceleration.<br>
Especially in reconstruction tasks of large scale scenes, our parrallel strategies shows stable adaption ability in accelerating the whole training process.<br>
To render with the parrallel model after training, using the command as the sequential model
```
python app/renderer.py --config confs/branch_parrallel_config_example.txt --ckpt=log/your_expname/your_expname.th
``` 
# 🤝 Authors
The main work comes from the LandMark Team, Shanghai AI Laboratory.<br>
Here are our honorable Contributors:

<a href="https://github.com/FlushingCat/LandMark-Documentation-Site/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=FlushingCat/LandMark-Documentation-Site" />
</a>