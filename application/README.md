# Experiment on Bitswap
I try to use bytes of video to construct images, then apply Bitswap on them.
So video can be compressed. After decompressing these images, we can convert these images into video.

Up to now I test 32x32 image, and train successfully with small resnet's depth.
However, official code `imagenet_compress.py` seems to support specific depth like 255 for 2 nz.
Limited by hardware, I cannot train such a big network thus cannot finish compress progress.

## Pesudo-image
Use `utils/processor.py` to slice and reconstruct video.
Before using, change `file_path` in it.
```shell
python utils/processor.py --mode 0  # video to image
python utils/processor.py --mode 1  # image to video
```

## Data preparation
Only for 32x32 imagenet format, can refer to official tutorial

Move pesudo-images into `bitswap/model/data/train_32x32` and `bitswap/model/data/valid_32x32`
then
```shell
cd bitswap/model/
python create_imagenet.py 
```

## Train
Only for 32x32 imagenet

ImageNet (32x32) (on 8 GPU's with OpenMPI + Horovod)
###### 4 latent layers
```
mpiexec -np 8 python imagenet_train.py --nz=4 --width=254
```
###### 2 latent layers
```
mpiexec -np 8 python imagenet_train.py --nz=2 --width=255
```
It seems that `--width` should be the same as command above or compression code cannot work


