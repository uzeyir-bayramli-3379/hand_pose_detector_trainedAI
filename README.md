# Hand Gesture Recognition with Trained Neural Network

A real-time hand gesture classifier built from scratch using MediaPipe for hand landmark detection and a small PyTorch neural network trained on hand-collected data. Recognizes custom gestures from a live webcam feed.

This project replaces the typical hardcoded-threshold approach to gesture detection (e.g. `if finger_distance < 0.075 ...`) with a learned classifier, making it trivial to add new gestures: just record more samples and retrain. No new detection logic required.

## Demo

The system runs end-to-end on a CPU at real-time framerates. It currently classifies three classes (configurable):
- **Class 0** — Vulcan salute
- **Class 1** — Peace sign
- **Class 2** — Negative / "other" class (anything that isn't a recognized gesture)

The "other" class is essential — it prevents the model from confidently misclassifying unrelated hand poses as one of the real gestures.

(Might be different due to slight changes in the testing - feel free to record your own data , very simple process)
## How It Works

The pipeline has four stages:

**1. Hand landmark extraction.** MediaPipe Hands detects a hand in the webcam frame and returns 21 3D landmarks (one per hand joint).

**2. Normalization.** Raw landmarks depend on hand position and size within the frame. To make the gesture *shape* the only thing the model sees, each sample is normalized by:
- Subtracting the wrist landmark from every point (centers the hand at the origin)
- Dividing every coordinate by the wrist-to-index-MCP distance (scales the hand to a consistent size)

This makes a given gesture look the same to the model whether the hand is near or far from the camera, or anywhere in the frame.

**3. Classification.** A small feedforward neural network takes the 63 normalized coordinates (21 landmarks × xyz) and outputs class scores. Architecture:

```
Input (63) → Linear → ReLU → Linear → Output (num_classes)
                64
```

Trained with cross-entropy loss and the Adam optimizer.

**4. Real-time inference.** The trained model runs on each frame's landmarks. Predictions below a configurable confidence threshold (default 0.75) are suppressed, so the system stays silent when unsure rather than guessing.

## Repository Structure

```
.
├── data_collection_from_camera.py   # Collect training data via webcam
├── model_training.py                # Train the classifier on collected data
├── real_time_model_usage.py         # Run the trained model live
├── vulcan_gesture_data.csv          # Collected training data (generated)
├── vulcan_gesture_model.pth         # Trained model weights (generated)
└── README.md
```

## Setup

Requires Python 3.9+.

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn torch
```

A webcam is required for both data collection and live inference.

## Usage

### 1. Collect Training Data

```bash
python data_collection_from_camera.py
```

A webcam window opens with MediaPipe overlaying hand landmarks. Controls:

| Key | Action |
|-----|--------|
| `r` | Toggle recording on/off |
| `n` | Advance to the next class label |
| `q` | Quit |

**Recommended workflow:**
1. Hold the first gesture (class 0). Press `r` to start recording. Move your hand around naturally — vary the angle, distance, and position slightly while keeping the gesture shape. Aim for ~300+ frames. Press `r` to stop.
2. Press `n` to advance to class 1. Repeat for the next gesture.
3. Press `n` again for class 2. **For the last class, record varied "other" poses** — open hand, fist, thumbs up, partial gestures, hand at rest. This teaches the model what *isn't* one of your real gestures.
4. Press `q` to save and exit.

Data is appended to `vulcan_gesture_data.csv` with one row per frame: 63 normalized coordinates + 1 label.

**Tips for good data:**
- Aim for roughly balanced class counts.
- The "other" class should contain **variety**, not consistency — many different non-gesture poses.
- Stop recording immediately if you make a wrong pose. Re-recording is cheap; bad labels are expensive.

### 2. Train the Model

```bash
python model_training.py
```

This reads `vulcan_gesture_data.csv`, splits it 80/20 into train/test, trains for 30 epochs, prints test accuracy, and saves the trained weights to `vulcan_gesture_model.pth`.

If you change the number of classes, update the model output layer (`nn.Linear(64, num_classes)`) in both `model_training.py` and `real_time_model_usage.py` to match.

Expected output: test accuracy in the 95–100% range for well-separated gestures with clean data.

### 3. Run Live Inference

```bash
python real_time_model_usage.py
```

Holds up gestures in front of the webcam; predictions appear on-screen when the model is confident (>0.75). Press `q` to quit.

## Implementation Notes

**Normalization at inference matches collection byte-for-byte.** The same wrist-centering and index-MCP scaling is applied in both scripts. This is the single most common source of bugs when deploying a trained classifier — train/inference preprocessing mismatch causes silent confident-but-wrong predictions.

**Confidence thresholding** prevents low-certainty predictions from displaying. The model still outputs a class for every input (`argmax` always picks one), but only predictions above 0.75 confidence are shown. Adjust the threshold in `real_time_model_usage.py` based on your tolerance for false positives vs missed detections.

**The "other" class matters.** A binary classifier trained only on "vulcan vs peace" will confidently call *any* hand pose one of those two — including unrelated gestures like an open wave. The negative class gives the model a place to put "five fingers extended but not vulcan," dramatically reducing false positives in real-world use.

## Adding New Gestures

This is the main payoff of using a learned model instead of hardcoded thresholds. To add a new gesture:

1. Run `data_collection_from_camera.py`, press `n` until you reach the next available class index, hold the new gesture, and record samples.
2. Increase the output layer size in both training and inference scripts to match the new class count.
3. Retrain.

No detection logic needs to change.

## Limitations and Known Issues

- Single-hand only. The current setup uses MediaPipe's default hand detection settings.
- Z-coordinate depth from MediaPipe is approximate and view-dependent; the model uses it but its quality varies.
- The dataset is small (~1000 samples for 3 classes) and reflects a single person's hands and lighting conditions. Adding more diverse training data would improve robustness across users and environments.
- The "other" class only generalizes as well as its variety. If you start seeing new false positives in the wild, record more of that specific pose into the "other" class and retrain.

## Tech Stack

- **MediaPipe Hands** — hand landmark detection
- **OpenCV** — webcam capture and display
- **PyTorch** — neural network and training loop
- **pandas / scikit-learn** — data loading and evaluation metrics
- **NumPy** — coordinate math
