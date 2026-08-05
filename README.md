# OpenCV Object Tracking: Benchmarking Classical Algorithms

This project benchmarks classical object tracking algorithms using the OTB2015 dataset. It provides both **Python** and **C++** implementations to compare performance across different environments.


## Algorithms Benchmarked
- **Python**: CSRT, KCF, MIL, MOSSE  
- **C++**: CSRT, KCF, MIL (MOSSE requires OpenCV contrib build, not included in default Ubuntu packages)


## Features
- Runs trackers across all OTB2015 sequences
- Computes:
  - Frames per second (FPS)
  - Mean Intersection-over-Union (IoU)
  - Robustness (failure rate)
- Supports multiple repeats per tracker
- Command-line arguments for flexible benchmarking
- Python version saves results to CSV for plotting success/precision curves
- **Interactive ROI demo in Python**: select a region of interest in a video and compare trackers live


## Requirements

### Python
- Python 3.8+
- OpenCV (`pip install opencv-contrib-python`)
- NumPy, Pandas, Matplotlib

### C++
- OpenCV 4.x (`libopencv-dev` on Ubuntu)
- C++17 compiler (g++ or clang++)
- CMake (optional, for easier builds)


## Usage

### Python Benchmark
```bash
python benchmark.py --dataset_root /path/to/OTB2015 --repeats 3 --tracker CSRT KCF MIL MOSSE
```

**Python Interactive ROI Demo**
```
python roi_demo.py
```

- Opens a video (default: Basketball.mp4 from OTB2015).

- Lets you draw a bounding box (ROI).

- Cycles through CSRT, KCF, MOSSE, MIL trackers.

- Displays FPS and bounding box overlay in real time.

- ESC key exits.

### C++ Benchmark
Compile:
``` bash
g++ main.cpp -o main `pkg-config --cflags --libs opencv4`
```

Run: 
``` bash
./main --dataset_root /path/to/OTB2015 --repeats 3 --tracker CSRT KCF MIL
```

Use --help to see options: 
```bash
./main --help
```

## Project structure
opencv-object-tracking/
│── python/
│   ├── benchmark.py              # Dataset-wide benchmarking
│   ├── interactive_roi_demo.py   # Interactive ROI demo
│   ├── utils.py                  # Helper functions
│   ├── requirements.txt          # Python dependencies
│   └── trackers/
│       ├── csrt_tracker.py
│       ├── kcf_tracker.py
│       ├── mil_tracker.py
│       └── mosse_tracker.py
│── cpp/
│   └── main.cpp                  # C++ benchmark implementation
└── README.md                     # Project documentation
