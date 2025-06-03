# Benchmarking ViT on Apple MLX vs. Nvidia GPU vs. Intel CPU

## Purpose 

## Model Selection 

I wanted to choose a model which was sophisticated, performs well on it's intended task but can run easily on consumer hardware. Here I selected the DeiT-small variant of the Vision Transformer (ViT), a patch based transformer architecture for image classification tasks. With 22M parameters it is sufficiently large to be realistic yet small enough to train on a Mac Mini or midrange RTX GPU. This architecture will remain consistent, with identical hyperparameters, in it's implementation across platforms. 

## Dataset 