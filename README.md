# Real-Time Hybrid Wildlife Tracking on Edge Networks

A deep learning-based wildlife tracking system that combines **YOLOv8** object detection with **Kalman Filter** motion prediction for robust multi-object tracking in video sequences.

---

## 📋 Background

Monitoring animal movements supports ecological research and wildlife conservation. Scientists need to understand how animals move, where they go, and how they behave to protect species and manage ecosystems.

Traditional methods like radio collars or GPS tags require catching animals and fitting them with devices, which is invasive, expensive, and not suitable for all species. Camera traps and surveillance cameras offer a non-intrusive way to monitor wildlife.

**Key Challenges:**
- Cameras are often in remote locations, running on batteries or solar power
- Connected to edge networks with limited bandwidth and computational resources
- Animals get hidden behind trees and move unpredictably
- Multiple animals might be present simultaneously
- System needs to handle these challenges while operating in real-time on edge devices

---

## 🎯 Objectives

Develop a video-based tracking system that can follow animals across frames, even when they are partially hidden or move unpredictably. The system should:

1. **Detect animals** in video frames
2. **Track individual animals** across frames
3. **Maintain identity** when animals temporarily disappear behind vegetation
4. **Handle multiple animals** without mixing up their identities
5. **Work efficiently** for field deployment scenarios

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **YOLOv8 Object Detection** | Utilizes YOLOv8 Nano (~3.2M parameters) for efficient real-time animal detection |
| **Kalman Filter Tracking** | Implements motion prediction for smooth trajectory estimation |
| **Hungarian Algorithm** | Optimal assignment between predictions and detections using IoU-based cost matrix |
| **Multi-Class Support** | Tracks multiple wildlife species (deer, horse, pig) |
| **Edge-Optimized** | Designed for deployment on edge devices with limited computational resources |

---

## 🛠️ Technology Stack

- **Python 3.11+**
- **Ultralytics YOLOv8** - State-of-the-art object detection
- **FilterPy** - Kalman filter implementation
- **OpenCV** - Computer vision operations
- **SciPy** - Hungarian algorithm (linear_sum_assignment)
- **NumPy & Pandas** - Data processing
- **Matplotlib & Seaborn** - Visualization

---

## 📁 Project Structure

```
Realtime-Hybrid-Wildlife-Tracking-on-Edge-Networks/
├── adsptest-fixed (2).ipynb       # Main implementation notebook
├── Wildlife_Tracking_Updated.pptx # Project presentation
└── README.md                      # This file
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy==1.26.4 scipy==1.13.1 ultralytics==8.3.49 filterpy==1.4.5 seaborn opencv-python pandas matplotlib
```

### Dataset Structure

The system expects the AnimalTrack dataset organized as:
```
Dataset/
├── Train/
│   ├── vedios/    # Training video files (.mp4, .avi, .mov)
│   └── gt/        # Ground truth annotations (.txt)
└── Test/
    ├── vedios/    # Test video files
    └── gt/        # Test ground truth files
```

### Usage

1. **Data Preparation**: Configure paths to your dataset
2. **Export to YOLO Format**: Convert annotations to YOLO format
3. **Train YOLOv8**: Fine-tune YOLOv8n on your wildlife dataset
4. **Run Tracking**: Apply the hybrid tracking pipeline

```python
# Key configuration
BASE_DIR = "/path/to/dataset"
TRAIN_VIDEO_DIR = os.path.join(BASE_DIR, "Train", "vedios")
TEST_VIDEO_DIR = os.path.join(BASE_DIR, "Test", "vedios")
```

---

## 🔬 Methodology

### Hybrid Tracking Approach

Combines motion prediction with appearance matching for robust real-time tracking. The advantages of using both approaches together include:
- Better handling of occlusions through motion prediction
- Reduced ID switches through appearance matching
- Improved robustness in challenging conditions

### 1. Object Detection (YOLOv8)
- **Model**: YOLOv8 Nano (smallest variant, ~3.2M params)
- **Input Resolution**: 640x640
- **Classes**: Deer, Horse, Pig
- **Training**: Fine-tuned on wildlife dataset

### 2. Kalman Filter for Motion Prediction

Kalman filters are recursive estimation algorithms that combine noisy position measurements with motion predictions to provide smooth, accurate tracking.

**State Vector**: `[x, y, s, r, dx, dy, ds]`
- `(x, y)`: Center coordinates
- `s`: Scale (area)
- `r`: Aspect ratio
- `dx, dy, ds`: Velocities

**Key Benefits:**
- Maintains estimate of animal's state (position, velocity)
- Updates with each new observation
- Robust to temporary occlusions or missed detections

### 3. Data Association (Hungarian Algorithm)
- IoU-based cost matrix between predicted and detected boxes
- Hungarian algorithm for optimal assignment
- Configurable IoU threshold for matching

### 4. Track Management
- **Track Initialization**: For new detections
- **Track Maintenance**: For matched detections
- **Track Termination**: For missed detections

---

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Sequences | 14 |
| Test Sequences | 5 |
| Classes | 3 (Deer, Horse, Pig) |
| Training Epochs | 20 |
| Image Size | 640x640 |
| Batch Size | 64 |

---

## 📈 Evaluation Metrics

The system evaluates tracking performance using:

### Detection Metrics
- **mAP50**: Mean Average Precision at IoU 0.5
- **mAP50-95**: Mean Average Precision across IoU thresholds
- **Precision & Recall**: Detection quality metrics

### Tracking Metrics
- **Position Accuracy**: How accurately animal positions are estimated
- **ID Switches**: How often animal identities are mixed up
- **Track Fragmentation**: How often tracks are lost and reacquired
- **Mostly Tracked (MT)**: Percentage of animals tracked for most of their time in view
- **Occlusion Handling**: Tracking maintenance during occlusions

---

## 📦 Datasets Used

| Dataset | Description |
|---------|-------------|
| **AnimalTrack Dataset** | Primary dataset with deer, horse, and pig sequences |

### Alternative Datasets for Extension
- WILDTRACK Dataset - Multi-camera pedestrian tracking
- Caltech Camera Traps Dataset - Large-scale camera trap data
- MOTChallenge Datasets - Multi-object tracking benchmarks
- VisDrone Dataset - Aerial tracking dataset
- Snapshot Serengeti - Camera trap project data

---

## 📋 Deliverables

1. ✅ **Working Implementation** - Video processing and animal tracking
2. ✅ **Visualizations** - Tracked animals with trajectories overlaid on video
3. ✅ **Presentation** - Project slides explaining the approach
4. ✅ **Metrics** - Tracking accuracy and stability measurements

---

## 🎯 Applications

- Wildlife conservation monitoring
- Animal behavior analysis
- Population studies
- Endangered species tracking
- Ecosystem health assessment

---

## 🔮 Future Extensions

- [ ] Classify different animal species automatically
- [ ] Recognize behaviors (grazing, running, resting)
- [ ] Estimate population sizes from tracking data
- [ ] Use multiple cameras for better coverage
- [ ] Predict animal movement patterns

---

## 📚 References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [SORT Algorithm](https://arxiv.org/abs/1602.00763)
- [FilterPy Library](https://github.com/rlabbe/filterpy)
- Bernardin & Stiefelhagen - CLEAR MOT Metrics
- OpenCV - Computer Vision Library

---

## 👥 Supervision

**Mini Project for Advanced Digital Signal Processing (ADSP)**

All projects are evaluated by **Dr. Upendra Kumar Sahoo** and coordinated by:
- TA Kannuru Srinadh
- Yerram Deekshith Kumar
- Debapriya Das Gupta

**ADSP Lab, NIT Rourkela**

---

## 📄 License

This project is for educational purposes.

---

*For questions or issues, please open an issue in the repository.*
