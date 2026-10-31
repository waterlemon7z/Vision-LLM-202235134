## 20261025 HW3
### File Structure
- hw3-1_gpu.py -> GPU 가속 (잔상효과, Convolution2D)
- hw3-2_MNIST.py -> MNIST 데이터셋 main_gpu() 함수
- hw3-3_CIFAR-10.py -> CIFAR-10 데이터셋 모델 학습 파트 + 저장한 모델 파일 (.pth)
- hw3-4_IoU.py -> IoU, NMS 구현

### How to run
``` bash
python3 {file}.py # In 20261025/ dir.
uv run {file}.py
```

### HW3-1 Output
- output/numpy_output.png
- output/cupy_output.png
``` bash
========== NumPy CPU ==========
Processing time: 8.0533 seconds

========== CuPy GPU ==========
CPU to GPU: 0.0635 seconds
GPU processing: 0.0011 seconds
GPU to CPU: 0.0005 seconds
Total GPU time: 0.0650 seconds

========== Comparison ==========
Compute-only speedup: 7568.19x
Including transfers: 123.82x

Process finished with exit code 0
```
![Ouptut](https://raw.githubusercontent.com/waterlemon7z/Vision-LLM-202235134/refs/heads/main/20261025/output/numpy_output.png)

### HW3-2 Output
``` bash
=== GPU Training ===
100.0%
100.0%
100.0%
100.0%
Epoch 1/3 | Train loss: 0.3133 | Train accuracy: 90.54% | Test loss: 0.0958 | Test accuracy: 96.96% | Time: 3.30s
Epoch 2/3 | Train loss: 0.0924 | Train accuracy: 97.15% | Test loss: 0.0694 | Test accuracy: 97.75% | Time: 3.23s
Epoch 3/3 | Train loss: 0.0663 | Train accuracy: 97.94% | Test loss: 0.0486 | Test accuracy: 98.29% | Time: 2.76s
Total CPU training time: 9.29s
GPU training time: 9.29s

Process finished with exit code 0
```
### HW3-3 Output
- output/CIFAR10_CNN.pth
``` bash
Epoch 1/5 | Train loss: 1.3086 | Train accuracy: 52.44% | Test loss: 1.4015 | Test accuracy: 54.85% | Time: 25.79s
Epoch 2/5 | Train loss: 0.8908 | Train accuracy: 68.51% | Test loss: 0.9000 | Test accuracy: 69.47% | Time: 21.89s
Epoch 3/5 | Train loss: 0.7244 | Train accuracy: 74.97% | Test loss: 0.7782 | Test accuracy: 73.51% | Time: 22.02s
Epoch 4/5 | Train loss: 0.6238 | Train accuracy: 78.43% | Test loss: 0.5721 | Test accuracy: 80.36% | Time: 21.97s
Epoch 5/5 | Train loss: 0.5570 | Train accuracy: 80.93% | Test loss: 0.5881 | Test accuracy: 80.45% | Time: 21.82s
Total GPU training time: 113.49s
```
### HW3-4 Output
![]()