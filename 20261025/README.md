## 20261017 HW2
### File Structure
 - only main.py

### How to run
``` bash
python3 main.py # In 20261025/ dir.
uv run main.py
```

### Output
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