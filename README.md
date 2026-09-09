# 🧠 Advanced Parkinson's Disease Diagnostic System

An end-to-end machine learning application built to assist in the early detection of Parkinson's Disease using biomedical voice measurements. This system implements an Artificial Neural Network (ANN) trained on the UCI Parkinson's Telemonitoring dataset, leveraging **balanced class weights** to overcome class imbalance and optimize minority class recall.

---

## 🚀 Key Features
* **Full-Feature 22-Parameter Input:** Captures precise acoustic metrics including fundamental frequency, jitter, shimmer, noise ratios, and nonlinear measures.
* **Balanced ANN Architecture:** Explicitly incorporates class weighting strategies during training to handle skewed class distributions and reduce false negatives for healthy/affected voice samples.
* **Interactive Gradio Web Interface:** Deployed with a clean, user-friendly UI allowing real-time diagnostic evaluation.
* **Research-Grade Evaluation:** Thoroughly benchmarked using confusion matrices and accuracy tracking comparing baseline models against balanced weight optimization.

---

## 📂 Dataset Overview
The model utilizes the **UCI Parkinson's Telemonitoring Dataset**, processing **22 acoustic features** extracted from voice recordings:
* **Frequency Parameters:** `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`
* **Jitter Parameters:** `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`
* **Shimmer Parameters:** `MDVP:Shimmer`, `MDVP:Shimmer(APQ3)`, `MDVP:Shimmer(APQ5)`, `MDVP:APQ`, `Shimmer:DDA`
* **Noise & Nonlinear Measures:** `NHR`, `HNR`, `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`

---

## ⚖️ Handling Class Imbalance with Balanced Weights
In medical datasets, healthy samples often outnumber disease cases (or vice-versa), leading to biased predictions. To resolve this:
* We integrated **computed class weights** (`class_weight='balanced'` style logic for the ANN training pipeline).
* This penalizes misclassifications of the minority class more heavily, ensuring the model maintains high sensitivity and reliable recall for accurate clinical screening indicators.

---

## 📊 Evaluation & Visual Artifacts
Visual proofs, performance metrics, and application outputs are securely stored in the `images/` directory for reporting and academic review:
* **Simple Model Accuracy:** Located at `images/simple_model_accuracy.png`
* **Balanced Model Accuracy:** Located at `images/balanced_model_accuracy.png`
* **Confusion Matrix Comparison:** Located at `images/confusion_matrix_comparison.png`
* **Healthy Voice UI Test:** Located at `images/gradio_healthy_prediction.png`
* **Parkinson's Detected UI Test:** Located at `images/gradio_parkinsons_prediction.png`

---

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Nokhiz-Khan/Parkinsons-Voice-Diagnostic-System.git](https://github.com/Nokhiz-Khan/Parkinsons-Voice-Diagnostic-System.git)
   cd Parkinsons-Voice-Diagnostic-System
Install dependencies:

Bash


pip install -r requirements.txt

Run the Gradio Application:

Python

python Parkinsons_Disease_Detection_ANN.ipynb

(A public link will be generated automatically to test the 22-feature interactive web interface).

🛠️ Tech Stack
Language: Python 3.x

Libraries: TensorFlow / Keras, Scikit-Learn, Pandas, NumPy

UI Framework: Gradio
