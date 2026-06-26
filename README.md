# Personalized Aesthetic Score Prediction of User Interfaces Using Federated Learning

![Federated Learning](https://img.shields.io/badge/Federated_Learning-Enabled-success)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-EfficientNet--B0-blue)
![Originality](https://img.shields.io/badge/Plagiarism-11%25-brightgreen)

## 📌 Abstract
Evaluating the aesthetics of user interfaces (UIs) is challenging because people's aesthetic sensibilities vary significantly based on their age, gender, occupation, and personal tastes. Most automated UI evaluation models produce a single average global score, ignoring individual differences. 

This project proposes a **Personalised Aesthetic Score Prediction System for User Interfaces using Federated Learning**. We collected a custom dataset of 100 desktop and mobile user interfaces across various domains, annotated with aesthetic ratings from 100 participants. The framework uses **EfficientNet-B0** and Federated Learning (FedAvg) to predict aesthetic scores while keeping user data local to each demographic group, ensuring privacy. We also developed an LLM-based aesthetic evaluation module using **Qwen3-VL-235B** for comparative analysis.

---

## 🌟 Key Features
- **Personalized Predictions:** Captures individual user preference deviations from the population mean to predict tailored aesthetic scores.
- **Privacy-Preserving:** Uses Federated Learning (FedAvg) so raw user ratings never leave the local device.
- **Custom UI Dataset:** Contains 100 UI screenshots (50 desktop, 50 mobile) from 10 diverse domains (education, news, OTT platforms, travel, sports, etc.).
- **Demographic Context:** Incorporates age, gender, and educational/occupational background into predictions.
- **LLM Benchmarking:** Features a zero-shot multimodal LLM (Qwen3-VL-235B) evaluation module for qualitative comparison.

---

## 🏗️ Proposed System Architecture

### 1. The Dataset
Since no existing dataset covers UI aesthetics with paired desktop and mobile screenshots alongside individual user ratings, we created a custom dataset:
- **100 UI Screenshots:** 50 desktop and 50 mobile versions.
- **10 Domains:** Government, Sports, Job Portals, News, OTT, Courier, Education, Travel, Environment, Shopping.
- **100 Participants:** Diverse age groups (6-58 years), genders, and occupational statuses (student, working, non-working).

### 2. Deep Learning Backbone: EfficientNet-B0
We experimented with EfficientNet-B0, ResNet-50, and ViT-B/16. **EfficientNet-B0** was chosen as the global model due to its stable performance, lightweight design (~5.3M parameters), and fast training time, making it highly suitable for federated environments.

### 3. Federated Learning Process
- **Data Distribution:** Distributed across 5 demographic clients (e.g., female_nonworking, male_student, etc.).
- **Process:** Each client trains the shared EfficientNet-B0 model locally. Only updated model weights are sent to the central server for aggregation using FedAvg.
- **Personalization:** Achieved via user-mean centering and min-max scaling, eliminating the need for complex local fine-tuning.

### 4. LLM-Based Evaluation Module
A standalone dashboard using **Qwen3-VL-235B** (via Ollama Cloud) to provide a zero-shot, prompt-based aesthetic assessment for benchmarking against the federated model.

---

## 📂 Repository Structure

```
.
├── Dataset/
│   ├── images/                        # UI Screenshots (Desktop & Mobile)
│   └── Aes_dataset - Sheet1.csv       # Participant ratings & demographics
├── FL training/
│   ├── federated-learning-1.ipynb     # Main notebook for FL training
│   └── results.zip                    # Saved model checkpoints and outputs
├── Global Model Select/
│   ├── EfficientNet/                  # Baseline training for EfficientNet-B0
│   ├── Resnet/                        # Baseline training for ResNet-50
│   └── Vit/                           # Baseline training for ViT-B/16
├── ui creation/
│   ├── fl-deploy.ipynb                # Gradio dashboard for Federated Model
│   └── llm-fl-deploy.ipynb            # Gradio dashboard for LLM comparison
├── FL(FINAL).pdf                      # Comprehensive Final Project Report
├── plag report.pdf                    # Plagiarism analysis (11% Overall Similarity)
└── README.md                          # Project Documentation
```

---

## 📊 Experimental Results

- **Model Performance:** The Federated EfficientNet-B0 achieved a validation **MSE of 0.0534**, **RMSE of 0.2312**, and **MAE of 0.1837** (on a normalized 0-1 scale).
- **Federated vs. Centralized:** The federated model matched the predictive accuracy of the centralized baseline exactly, proving that privacy-preserving distributed training did not degrade performance.
- **Originality:** The project report has been verified with an **11% overall similarity index** in the plagiarism report, highlighting the original contribution of this research.

---

## 🛠️ Software Tools & Platforms
- **Language:** Python 3.12
- **Deep Learning:** PyTorch 2.10, torchvision
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib
- **Web UI:** Gradio 5.x
- **LLM Integration:** Ollama (Python client) for Qwen3-VL-235B
- **Hardware:** NVIDIA Tesla T4 GPUs (via Kaggle)

---

## 🔮 Future Work
- Integrating advanced federated strategies like FedProx or Per-FedAvg to improve extreme rating predictions.
- Deploying the federated framework in a real distributed network to evaluate latency and asynchronous training dynamics.
- Expanding the dataset with more diverse interfaces and cultural backgrounds.
