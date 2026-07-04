# Face Tracker & Hands-Free Cursor Controller

A hands-free, machine learning-powered cursor controller that translates your facial movements and gestures into computer inputs. It utilizes MediaPipe for facial landmark and hand tracking, and a custom Graph Neural Network (GNN) combined with a Physics-Informed Neural Network (PINN) built in PyTorch to smoothly translate head movements into precise cursor controls.

## Features

- **Face Cursor Tracking**: Move your head to control the mouse cursor seamlessly on the screen. The custom GNN+PINN model ensures smooth tracking and nonlinear acceleration.
- **Mouth Click**: Open your mouth to trigger left and right mouse clicks.
- **Scroll Mode**: Scroll vertically by shifting your head up and down.
- **Gesture Controls**:
  - **Volume Control**: Pinch with your right hand and move your fingers to adjust system volume.
  - **Brightness Control**: Pinch with your left hand to adjust screen brightness.
- **Fitts' Law Evaluation**: Includes a custom Pygame-based environment to scientifically evaluate cursor performance (throughput and error rate) using Fitts' Law.

## Requirements

Ensure you have Python 3.8+ installed. The following libraries are required:

```bash
pip install opencv-python mediapipe torch torch-geometric pyautogui screen-brightness-control pycaw pygame numpy
```
*(Note: You may need to install `torch` and `torch-geometric` using specific commands based on your CUDA version. Refer to the [PyTorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) documentation.)*

## Usage

### 1. Data Collection (Optional)
If you wish to train your own model instead of using the provided weights, you can collect your own head movement data.
```bash
python collect_data.py
```
This will open your webcam. Move your head naturally to generate samples. Press `Q` to quit and save the data to `data/samples.npz`.

### 2. Training the Model (Optional)
Train the custom GNN + PINN architecture on your collected data.
```bash
python train.py
```
This saves the trained weights to `models/face_cursor.pth`.

### 3. Running the Face Cursor
Start the main controller application:
```bash
python run.py
```
**Controls during runtime:**
- `E` - Toggle Cursor Mode (Head tracking)
- `S` - Toggle Scroll Mode
- `V` - Toggle Volume Control (Right hand pinch)
- `B` - Toggle Brightness Control (Left hand pinch)
- `D` - Disable all modes
- `Q` - Quit the application

### 4. Evaluating with Fitts' Law
To test the usability and precision of the face cursor, run the Fitts' Law test environment:
```bash
python fitts_eval/fitts_test.py
```
Once you complete the trials, you can analyze your performance (throughput, movement time, and error rate):
```bash
python fitts_eval/analyze_fitts.py
```

## Architecture

- **MediaPipe Face Mesh & Hands**: Extracts real-time 3D coordinates for facial and hand landmarks.
- **FaceGNN (`models/gnn.py`)**: A Graph Convolutional Network that processes the complex spatial relationships of the facial landmarks to infer intended direction.
- **MotionPINN (`models/pinn.py`)**: A Physics-Informed neural layer designed to smooth out the positional jitter and mimic natural kinematics.
- **PyAutoGUI & Pycaw**: Used to interface with the operating system for simulated mouse movements, clicks, and system-level audio adjustment.


