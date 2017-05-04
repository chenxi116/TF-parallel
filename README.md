# TF-parallel

Authors: Shijie Wu, Chenxi Liu, Siyuan Qiao

### Installation

- Tensorflow 1.1 
- Tensorflow `models` repository
```
cd TF-parallel
git clone https://github.com/tensorflow/models/
```
- Download `inception_v4` model file
```
cd TF-parallel
mkdir checkpoints
cd checkpoints
wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
tar -xvf inception_v4_2016_09_09.tar.gz
```
- Download ImageNet validation set and link to `imagenet` folder
```
cd TF-parallel/imagenet
ln -s /your/path/to/ILSVRILSVRC2012_img_val/ ./
```

### Running
```
python serial-one.py 0
python serial-imagenet-val.py 0
```
The `0` is the GPU_ID.

### Mode Filtering and Parallel Image Fetching
```
python parallel-imagenet-blur-val.py 0 32
```
The `0` is the GPU_ID and the `32` is the number of threads.
