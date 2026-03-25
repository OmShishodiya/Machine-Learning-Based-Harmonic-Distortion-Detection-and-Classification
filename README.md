⚡ Machine Learning-Based Detection and Classification of Harmonic Distortions
📌 Domain

Energy | Data Science | Machine Learning | Signal Processing | Power Systems

🚀 Overview

This project presents a machine learning-driven framework for detecting and classifying harmonic distortions in electrical power systems.

Harmonic distortions degrade power quality, reduce efficiency, and can damage equipment. Traditional methods rely heavily on manual analysis and fixed thresholds.

👉 This system automates the process using:

Signal processing techniques for harmonic feature extraction
Machine learning models for classification and pattern recognition
Advanced reconstruction methods for distortion analysis and correction
🎯 Objective

The goal is to develop a robust system that can:

Simulate and/or process electrical signals containing various harmonic distortions
Extract key harmonic features such as:
Total Harmonic Distortion (THD)
Individual harmonic magnitudes
Statistical signal features
Train ML models to:
Classify distortion levels
Identify dominant harmonic components
Evaluate performance using:
Accuracy
Precision
Recall
Robustness under noisy conditions
Demonstrate the superiority of ML-based approaches over traditional harmonic analysis
🧠 Key Concepts
Harmonic Analysis
Power Quality Monitoring
Time-Series Signal Processing
Frequency Domain Analysis (FFT)
Machine Learning Classification
Signal Reconstruction & Denoising
📂 Dataset
CSV-based dataset of power system signals
Each row represents a signal waveform sample
Columns represent amplitude values over time
Includes signals with varying:
Harmonic orders
Distortion levels
Noise conditions
⚙️ System Pipeline
Raw Power Signal
        ↓
Preprocessing (Normalization, Cleaning)
        ↓
Feature Extraction
   - FFT
   - THD Calculation
   - Statistical Features
        ↓
Model Training & Testing
        ↓
Classification (Distortion Level / Harmonic Type)
        ↓
Reconstruction & Distortion Analysis
        ↓
Visualization & Evaluation
🔍 Feature Extraction

Key features used in this project:

Total Harmonic Distortion (THD)
Harmonic amplitude spectrum (via FFT)
Signal energy
Mean, variance, skewness
Peak and envelope characteristics
🤖 Models Implemented

Each model is implemented in a separate Python script for independent execution and comparison.

🔹 Classical ML Models
Support Vector Machine (SVM)
Decision Tree
Random Forest
Isolation Forest
One-Class SVM
🔹 Deep Learning Models
Autoencoder
Denoising Autoencoder
1D CNN
LSTM
🔹 Advanced / Research Models
Physics-Informed Neural Network (PINN)
Neural ODE
Fourier Neural Operator (FNO)
GAN (Signal Denoising)
Diffusion Model
🔹 Signal Processing Techniques
Kalman Filter
Wiener Filter
FFT-based analysis
⚡ Distortion Analysis & Correction

The system not only detects distortion but also estimates and corrects it:

distortion = original_signal - reconstructed_signal
anti_wave = -distortion
corrected_signal = original_signal + anti_wave

👉 This mimics active harmonic compensation used in modern power systems.

📊 Evaluation Metrics

Models are evaluated using:

Accuracy
Precision
Recall
F1 Score
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
📈 Visualizations
Time-domain signal comparison
Frequency spectrum (FFT plots)
Spectrograms
Reconstruction error plots
Confusion matrix
Error distribution histograms
🛠️ Tech Stack
Python
NumPy, Pandas
Matplotlib
Scikit-learn
TensorFlow / PyTorch
SciPy
▶️ How to Run
1. Clone repository
git clone https://github.com/your-username/harmonic-distortion-ml.git
cd harmonic-distortion-ml
2. Install dependencies
pip install -r requirements.txt
3. Run models individually
python autoencoder.py
python svm_model.py
python random_forest.py
python kalman_filter.py
🧪 Experimental Strategy
Each model is executed independently
Performance is compared across:
Accuracy
Reconstruction quality
Noise robustness
Visualization used for qualitative analysis
🧠 Key Insights
ML models outperform traditional threshold-based harmonic detection
Frequency-domain features significantly improve classification
Reconstruction-based models (autoencoders) provide deeper distortion understanding
Hybrid approaches (Signal Processing + ML) yield the best results
🔮 Future Work
Real-time deployment in smart grids
Integration with IoT-based power monitoring systems
Hardware implementation (DSP/FPGA)
Adaptive learning for dynamic grid conditions
Hybrid PINN-based physics-aware models
🤝 Contributions

Contributions are welcome!
Feel free to fork the repository and submit pull requests.

📜 License

MIT License

⭐ Final Note

This project bridges the gap between:

⚡ Power Systems Engineering
🤖 Machine Learning
📡 Signal Processing

and demonstrates how intelligent systems can improve power quality monitoring and automation.
