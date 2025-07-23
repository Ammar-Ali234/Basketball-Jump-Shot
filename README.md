# 🏀 AI Basketball Shot Detection

An AI-powered system for detecting and analyzing basketball jump shots using real-time pose estimation and object detection.
https://youtu.be/fb1L5V_qGEI

---

## 🎥 Demo Video

[![Watch the Demo](https://img.youtube.com/vi/fb1L5V_qGEI/0.jpg)](https://youtu.be/fb1L5V_qGEI)

---

## 🙋‍♂️ Want Help?

If you're interested in collaborating, improving the system, or just have questions:

- 📧 Email: [mammarali299@gmail.com](mailto:mammarali299@gmail.com)  
- 💼 LinkedIn: [linkedin.com/in/mammarali](https://www.linkedin.com/in/mammarali)


## 📌 Features

- 🔍 **Real-time Shot Detection** using MediaPipe and YOLOv8
- 🧍‍♂️ **Pose Estimation**: Recognize shot phases (stance, jump, release, landing)
- 🏀 **Ball & Rim Tracking**: Detects basketball and hoop using YOLO
- 🎯 **Made/Miss Classification**: Based on ball–rim bounding box overlap
- 🧠 **Smart Logic**: Waits after release to accurately judge shots
- 🎥 **Annotated Video Output** with trajectory lines and shot result (`✅ SCORED` or `❌ MISS`)
- 📝 **Performance Logging** to a `.txt` file

---


## 🛠️ Tech Stack

| Component       | Tool/Library         |
|----------------|----------------------|
| Pose Estimation| [MediaPipe](https://github.com/google/mediapipe) |
| Object Detection| [YOLOv8](https://github.com/ultralytics/ultralytics) |
| Video Processing| OpenCV               |
| Language        | Python               |

---

## 🚀 How It Works

1. **Pose landmarks** are detected for the player using MediaPipe.
2. **YOLOv8** identifies the ball, rim, and player.
3. Phases of the shot are detected:
   - **Stance**  
   - **Jump**  
   - **Release**  
   - **Landing**
4. After release, the script waits to see if the **ball's bounding box touches the rim's**.
5. If it overlaps, it's counted as a **made shot**, otherwise it's marked a **miss** after a short timeout.
6. Output includes:
   - Total shots
   - Made/missed counters
   - Visual trajectory paths (green = made, red = missed)
   - An `end_moo.mp4` video and a `release_info.txt` summary file

---

## 📦 Installation

```bash
git clone https://github.com/Ammar-Ali234/Basketball-Jump-Shot.git
cd basketball-shot-detector

# Install dependencies
pip install -r requirements.txt
```

Make sure you have:
- Python 3.8+
- OpenCV
- MediaPipe
- Ultralytics (YOLOv8)

---

## ▶️ Usage

1. Place your input video in the project folder (e.g., `07222.mov`)
2. Run the main script:

```bash
python main.py
```

The output video will be saved as `end_moo.mp4`, and logs to `release_info.txt`.

---


## 🧪 Future Features (Coming Soon)

- Shooting angle analysis (elbow, wrist, knee)
- Heatmaps and accuracy graphs
- Sound or voice feedback
- Multi-player tracking support

---

