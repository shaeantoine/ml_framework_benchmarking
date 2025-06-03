# Benchmarking ViT on Apple MLX vs. Nvidia GPU vs. Intel CPU

## Purpose 

## Model Selection 

I wanted to choose a model which was sophisticated, performs well on it's intended task but can run easily on consumer hardware. Here I selected the DeiT-small variant of the Vision Transformer (ViT), a patch based transformer architecture for image classification tasks. With 22M parameters it is sufficiently large to be realistic yet small enough to train on a Mac Mini or midrange RTX GPU. This architecture will remain consistent, with identical hyperparameters, in it's implementation across platforms. 

## Dataset 

I've selected [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) to train each of the architectures. It consists of 100 classess with 600 images each. These images are originally sized at 32x32 so I'll need to scale them to conform to the 224x224 shape DeiT-small ViT expects. 

## Hardware and Frameworks 

### Apple Silicon - MLX
An M4 Mac Mini will be used to assess the performance of Apple Silicon. The M series chips include a CPU, GPU and Neural Engine. A core feature is the unified memory which means there is no need for explicit data transfer between the CPU and GPU. The MLX framework utilizes this feature optimize ML tasks.

### Nvidia GPU - TensorFlow (with CUDA Enabled)
A mid tier RTX 30/40 series card will be used to assess consumer Nvidia performance. TensorFlow will be used here to take advantage of CUDA which is the standard setup for GPU training. 

### Intel CPU - PyTorch
A recent multicore Intel CPU wil be used to assess consumer Intel performance. PyTorch will be used here with Intel's extensions for CPU acceleration. 

**A Note on OS:** *Obviously the OS can't be kept consistent for each device without serious effort. As a result, MacOS will be run for Apple Silicon and Linux will be used for Nvidia and Intel.*

## Performance Metrics

These metrics follow ML benchmarking guidance such as [DAWNBench](https://www.google.com/search?client=safari&rls=en&q=DAWNBench&ie=UTF-8&oe=UTF-8). In addition to throughput and inference speed markers. 

### Training Throughput 
Number of training samples processed per second (averaged over full training period). Generally the higher the better (training should be completed faster). 

### Inference Latency
Time per image (ms) for a forward pass. This correlates to deployment performance. I'll report the median and 90th percentile of a single-image. 

### Peak Memory Usage
Maximum device memory used during training (in GB). 

### Time-to-convergence
Time to reach a target validation accuracy. 

### Cost per training run 
This is specifically for cloud costs, I'll be tracking total cost for training. 

### Power Consumption 
Eletrcial power draw (in Watts) during training. 


## Methodology

### Batch Sizes and Workloads
I'll first test to see the largest batch that fits within the device's memory (maximum feasible batch). But I'll be primarily reporting results from a common reference batch (i.e. 32). 

I'll also be measuring latency at inference time with a batch size of 1. 

### Training Procedure 
I'll be using an identical model, optimizer and data pipeline implemented in each framework. I'll also be running either for a fixed number of epochs or until I hit a convergence threshold.

#### Metrics Measurement during Training
I'll be measuring throughput with average number of images processed per time incremement. I'll measure time-to-convergence by measuring how many seconds/ epochs each platform takes to reach a convergence threshold. I'll be monitoring memory usage through different tools on each platform and power measurement also through OS specific tools.

### Inference Procedure

#### Latency vs Throughput
I'll be measuring time to run a single forward pass (with batch=1) on new inputs. While also measuring the number of images/sec for larger batched.

### Replication and Consistency 
I'll run training and inference on each hardware multiple times to reduce variability. However, the inevitable variance in OS and potential difference in implementation could invite confounding variables. 