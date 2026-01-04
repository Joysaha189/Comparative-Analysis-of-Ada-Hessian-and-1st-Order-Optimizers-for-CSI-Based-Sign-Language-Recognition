# Comparative Analysis of Ada-Hessian and 1st-Order Optimizers for CSI-Based Sign Language Recognition

This repository contains the **implementation and experimental analysis of Ada-Hessian**, a second-order optimization method, compared against several **first-order optimizers** for **CSI-based sign language recognition**.  
The project was completed for **AMAT 591: Optimization Methods and Nonlinear Programming (Spring 2025)** at the **University at Albany**.

---

## 📌 Project Overview

First-order optimizers like **SGD** and **Adam** dominate deep learning due to efficiency but are often sensitive to hyperparameters. This project investigates whether using **second-order curvature information** via **Ada-Hessian** improves convergence speed, stability, and robustness when training neural networks on **Wi-Fi CSI-based sign language data**.

### Study Includes

- Implementation of **Ada-Hessian** using **Hutchinson’s method** for Hessian diagonal approximation  
- **Spatial (block-wise) averaging** to reduce stochastic curvature noise  
- Comparison with **SGD, Adam, AdamW, Adamax, Nadam, RMSprop**  
- Evaluation on multiple CSI datasets, including single-user and multi-user scenarios  

---

## 📁 Repository Structure





```

├── Code/

│   ├── adahessian\_sample\_usage          # Sample code for demonstration.

│

├── Data/

│   └── Home/                            # Subset of the Home CSI dataset (for demonstration)

│

├── Results/

│   ├── plots/                           # Training/validation curves

│

├── Deliverables/

│   ├── Project Report.pdf               # Final project report

│   ├── AMAT\_591\_Project\_Proposal.pdf    # Initial project proposal

│   └── Poster\_Joy.pdf                   # Project poster

│

├── Materials/

│   ├── signfi\_paper.pdf                 # Sign-Fi dataset reference paper

│   └── adahessian\_paper.pdf             # Ada-Hessian original paper

│   └── FuseLoc.pdf             # Paper Used for Phase Preprocessing
 
└── README.md

```






---

## 🧠 Problem Formulation

The learning objective is to minimize a non-convex empirical risk function:

\[
\min_{\theta} L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(x_i, y_i; \theta)
\]

![adhessian](Images/trajectory.png)

Unlike first-order methods, **Ada-Hessian** preconditions gradients using an approximate inverse Hessian, enabling curvature-aware updates that adapt to the loss surface geometry.

---

## ⚙️ Methodology

### Ada-Hessian Key Components

- Hessian diagonal approximation via **Hutchinson’s method**  
- Block-wise spatial averaging of curvature estimates  
- Momentum-based smoothing (similar to Adam)  
- Tunable Hessian power parameter (k) to interpolate between gradient descent and Newton-like updates  

### Neural Network Architecture

![cnn](Images/cnn.png)

- CNN-based classifier for CSI tensors of shape `(200 × 60 × 3)`  
- Convolution + BatchNorm + ReLU  
- Average pooling and dropout  
- Fully connected layer with softmax activation  

---

## 📊 Experimental Setup

### CSI Datasets

| Dataset | # Signs | Repetitions | # Instances |
| ------- | ------- | ----------- | ----------- |
| Home    | 276     | 10          | 2,760       |
| Lab     | 276     | 20          | 5,520       |
| Lab150  | 150     | 10          | 7,500       |

> ⚠️ Due to size constraints, only a subset of the Home dataset is included in `Data/`.

### Training Configuration

- Batch size: 256  
- Epochs: up to 300 (Ada-Hessian typically converges within ~50 epochs)  
- Weight decay: \(5 \times 10^{-4}\)  
- Learning rate:  
  - First-order optimizers: 0.01  
  - Ada-Hessian: 0.15  
- Learning rate decay at epochs 80, 160, 240  

---

## 📈 Results Summary

- Ada-Hessian converges significantly faster than first-order optimizers  
- Smooth and stable training behavior  
- Highest validation accuracy on the Lab dataset  
- Competitive performance with **AdamW**, the strongest first-order baseline  

![results](Images/results.png)

Detailed results for all configurations are available in the `Results/` directory.

---

## ⏱️ Computational Trade-offs

- Ada-Hessian requires 3–5× higher per-epoch training time due to Hessian estimation  
- Fewer epochs needed to converge, reducing overall tuning effort  
- Particularly effective in high-variability and multi-user CSI scenarios  

---

## ✅ Conclusions

- Ada-Hessian is robust and efficient for CSI-based sign language recognition  
- Less sensitive to learning rate selection than first-order methods  
- Weight decay remains an important hyperparameter  
- Demonstrates practical benefits of second-order information in real-world deep learning tasks  

---

## 📚 References

1. Yongsen Ma, Gang Zhou, Shuangquan Wang, Hongyang Zhao, and Woosub Jung. *SignFi: Sign language recognition using Wi-Fi.* Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., 2(1), March 2018.  
2. Zhewei Yao, Amir Gholami, Sheng Shen, Kurt Keutzer, Michael W. Mahoney. *AdaHessian: An adaptive second order optimizer for machine learning.* AAAI, 2021.  
3. T. F. Sanam and H. Godrich. *FuseLoc: A CCA Based Information Fusion for Indoor Localization Using CSI Phase and Amplitude of WiFi Signals.* ICASSP 2019, Brighton, UK, pp. 7565–7569.  

---

## Project Status

✅ **Completed** — Baseline implementation  

🔧 **Open for enhancements and upgrades**

---

## Acknowledgements

The initial components, including CSI data preprocessing and baseline pipeline, were carried out during undergraduate research. Special thanks to:

- **Dr. Hafiz Imtiaz** and **Dr. Tahsina Farah Sanam** for guidance and foundational contributions  
- **Dr. Zi Yang** for valuable guidance and feedback  

Implementation references:

- [AdaHessian Original Implementation](https://github.com/amirgholami/adahessian)  
- [SignFi Dataset & Method](https://yongsen.github.io/SignFi/)  

---

## Author

**Joy Saha**  
University at Albany, SUNY  

---

## License

This project is for **academic and educational purposes only**.






