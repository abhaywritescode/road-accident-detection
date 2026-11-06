# 🚨 Road Accident Detection via IMU Telemetry

An end-to-end machine learning pipeline that identifies vehicle collisions and anomalies using high-frequency smartphone inertial measurement unit (IMU) telemetry (accelerometer, gyroscope, and speed data).

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

Rapid post-crash detection saves lives by reducing emergency response times. This project implements a lightweight classification model trained on empirical smartphone sensor data to classify crash events.

> **Note on Model Selection:** While initial experimentation considered basic linear approaches, this pipeline uses **Logistic Regression**—a linear decision boundary classifier optimized for binary outcome probability (Probability of Crash = 1) rather than continuous regression.

---

## 📊 Dataset

Trained on the **[Smartphone IMU Road Accident Detection Dataset](https://www.kaggle.com/datasets/drabdulbari/smartphone-imu-road-accident-detection-dataset)**, containing synchronized kinematic readings:

- **Kinematics:** Linear acceleration (`Motion_Intensity`), vehicle velocity (`Speed_kmh`).

- **Rotational Dynamics:** 3-axis gyroscope telemetry (`Gyro_X`, `Gyro_Y`, `Gyro_Z`).

- **Ground Truth:** Binary crash indicator `Crash_Label` (`0 = Normal Event`, `1 = Accident Event`).


---

## ⚙️ Data Processing & Feature Engineering

1. **Cleaning:**

   * Handled missing records (`dropna`) and stripped redundant observations (`drop_duplicates`).
   * Dropped non-kinematic spatial and temporal features (`Timestamp`, `Latitude`, `Longitude`) to prevent spatial overfitting.

2. **Unit Normalization:**
   * Transformed vehicle speed from km/h to SI metric m/s:
     `Speed (m/s) = Speed (km/h) / 3.6`

3. **Rotational Invariance (Feature Engineering):**
   * Engineered **`Angular_Speed`** (rad/s) using the Euclidean L2 norm across all three rotational axes to capture aggregate rotational energy regardless of phone orientation:
     `Angular_Speed = sqrt(Gyro_X^2 + Gyro_Y^2 + Gyro_Z^2)`

4. **Standardization:**
   * Features are centered to zero-mean and unit-variance using Scikit-Learn’s `StandardScaler`:
     `z = (x - mean) / standard_deviation`

---

## 🧠 Model Architecture & Pipeline

```text
Raw IMU Data (Acc, Gyro, Speed)
   │
   ▼
Cleaning & Spatial Column Drop
   │
   ▼
Feature Engineering (Euclidean Angular Speed)
   │
   ▼
StandardScaler Normalization
   │
   ▼
Logistic Regression Classifier ──► P(Crash) >= 0.5 ──► [Normal | Accident]

```

### Validation Metrics

The dataset is partitioned using an **80/20 train-test split** (`random_state=42`). Performance is evaluated on:

* **Accuracy:** Overall correctness across all event classes.
* **Macro Precision:** Minimizing false alarm triggers.
* **Binary F1-Score:** Harmonic mean of precision and recall on positive collision events (Class = 1).
* **Confusion Matrix:** Analysis of false positives vs. critical false negatives.
* **Explainability:** Direct inspection of model weights (w) and bias (b) to quantify feature influence on crash probability.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository:**

```bash
git clone [https://github.com/your-username/road-accident-detection.git](https://github.com/your-username/road-accident-detection.git)
cd road-accident-detection

```

2. **Install dependencies:**

```bash
pip install -r requirements.txt

```

3. **Run the Notebook:**

```bash
jupyter notebook notebook.ipynb

```

---

## 📂 Project Structure

```text
road-accident-detection/
├── dataset.csv
├── notebook.ipynb
├── requirements.txt
├── README.md
└── LICENSE

```

---

## 🔮 Future Roadmap

* [ ] **Android Native Deployment:** Implement the Logistic Regression math `y = sigmoid(w*x + b)` directly in **Kotlin** using the extracted model weights and bias. This avoids heavy ML frameworks and allows the model to run continuously as an ultra-lightweight background service in a native Android application.
* [ ] **Embedded Hardware Integration:** Hardcode the model weights into edge microcontrollers (**ESP32 / Arduino / STM32**) coupled with dedicated MEMS sensors (**MPU-6050 / BMI160**) for real-time, zero-dependency inference on low-compute hardware.
* [ ] **Virtual Crash Simulation:** Build a synthetic kinematic simulation environment (via physics engines or CARLA simulator) to evaluate edge-case collision vectors without physical testing.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

```
