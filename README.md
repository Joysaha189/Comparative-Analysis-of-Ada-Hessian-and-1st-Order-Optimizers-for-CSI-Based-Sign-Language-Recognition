# Comparative-Analysis-of-Ada-Hessian-and-1st-Order-Optimizers-for-CSI-Based-Sign-Language-Recognition



This repository contains the implementation and experimental analysis of ***Ada-Hessian***, a second-order optimization method, compared against several **first-order optimizers** for **CSI-based sign language recognition**.

The project was completed for \*\*AMAT 591: Optimization Methods and Nonlinear Programming (Spring 2025)\*\* at the \*\*University at Albany\*\*.





###### **📌 Project Overview**



First-order optimizers such as SGD and Adam dominate deep learning due to their efficiency but are often highly sensitive to hyperparameter choices. This project investigates whether incorporating \*\*second-order curvature information\*\* via ***Ada-Hessian*** can improve convergence speed, stability, and robustness when training neural networks on **WiFi CSI-based sign language data**.



**The study includes:**



* An implementation of **Ada-Hessian using Hutchinson’s method** for Hessian diagonal approximation
* **Spatial (block-wise) averaging** to reduce stochastic curvature noise
* Extensive comparison with **SGD, Adam, AdamW, Adamax, Nadam, and RMSprop**.
* Evaluation on multiple CSI datasets, including single-user and multi-user settings





###### &nbsp;📁 Repository Structure



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

├── deliverables/

│   ├── Project Report.pdf               # Final project report

│   ├── AMAT\_591\_Project\_Proposal.pdf    # Initial project proposal

│   └── Poster\_Joy.pdf                   # Project poster

│

├── Materials/

│   ├── signfi\_paper.pdf                 # Sign-Fi dataset reference paper

│   └── adahessian\_paper.pdf             # Ada-Hessian original paper

│

└── README.md

```





\## 🧠 Problem Formulation



The learning objective is to minimize a non-convex empirical risk function:



\[

\\min\_{\\theta} ; L(\\theta) = \\frac{1}{N}\\sum\_{i=1}^{N} \\ell(x\_i, y\_i; \\theta)

]



While first-order methods rely only on gradient statistics, \*\*Ada-Hessian preconditions gradients using an approximate inverse Hessian\*\*, enabling curvature-aware updates that adapt to the geometry of the loss surface.





\## ⚙️ Methodology



\### Ada-Hessian Key Components



\* \*\*Hessian diagonal approximation\*\* via Hutchinson’s method

\* \*\*Block-wise spatial averaging\*\* of curvature estimates

\* \*\*Momentum-based smoothing\*\*, similar to Adam

\* Tunable \*\*Hessian power parameter (k)\*\* to interpolate between gradient descent and Newton-like behavior



\### Neural Network Architecture



\* CNN-based classifier for CSI tensors of shape `(200 × 60 × 3)`

\* Convolution + BatchNorm + ReLU

\* Average pooling and dropout

\* Fully connected layer with softmax activation



---



\## 📊 Experimental Setup



\### CSI Datasets



| Dataset | # Signs | Repetitions | # Instances |

| ------- | ------- | ----------- | ----------- |

| Home    | 276     | 10          | 2,760       |

| Lab     | 276     | 20          | 5,520       |

| Lab150  | 150     | 10          | 7,500       |



> ⚠️ Due to size constraints, \*\*only a subset of the Home dataset\*\* is included in this repository under `data/`.



---



\### Training Configuration



\* Batch size: 256

\* Epochs: up to 300 (Ada-Hessian typically converges within ~50 epochs)

\* Weight decay: (5 \\times 10^{-4})

\* Learning rate:



&nbsp; \* First-order optimizers: 0.01

&nbsp; \* Ada-Hessian: 0.15

\* Learning rate decay at epochs 80, 160, and 240



---



\## 📈 Results Summary



\* \*\*Ada-Hessian converges significantly faster\*\* than first-order optimizers

\* Demonstrates \*\*smooth and stable training behavior\*\*

\* Achieves \*\*highest validation accuracy on the Lab dataset\*\*

\* Performance is competitive with \*\*AdamW\*\*, the strongest first-order baseline



Detailed results for \*\*all configurations\*\* (learning rates, Hessian power values, weight decay settings) are available in the `results/` directory.



---



\## ⏱️ Computational Trade-offs



\* Ada-Hessian incurs \*\*3–5× higher training time\*\* due to Hessian estimation

\* However, it \*\*requires far fewer epochs to converge\*\*, reducing tuning effort

\* Particularly effective in \*\*high-variability and multi-user CSI settings\*\*



---



\## ✅ Conclusions



\* Ada-Hessian is a robust and efficient optimizer for CSI-based sign language recognition

\* Less sensitive to learning rate selection than first-order methods

\* Weight decay remains an important hyperparameter

\* Demonstrates the practical benefits of second-order information in real-world deep learning tasks



---



\## 📚 References



1\. \*\*Ma et al.\*\* \*SignFi: Sign Language Recognition Using WiFi\*. ACM IMWUT, 2018.

2\. \*\*Yao et al.\*\* \*AdaHessian: An Adaptive Second Order Optimizer for Machine Learning\*. AAAI, 2021.



---



\## 🙏 Acknowledgments



\* SignFi dataset and prior CSI-based sign language recognition research

\* \*\*Dr. Zi Yang\*\*, AMAT 591

\* \*\*Dr. Hafiz Imtiaz\*\* and \*\*Dr. Tahsina Farah Sanam\*\* for earlier guidance during undergraduate research



---



If you want, I can next:



\* Add a \*\*Quick Start / How to Run\*\* section

\* Create a \*\*minimal README\*\* version for public release

\* Help you write a \*\*GitHub release description or project tagline\*\*



\#### Project Status







!\[results](Images/final\\\_results.png)







✅ Completed — Baseline implementation







🔧 Open for enhancements and upgrades











\###### \*\*Acknowledgements\*\*







Special thanks to \*\*Dr. Hafiz Imtiaz\*\* and \*\*Dr. Tahsina Farah Sanam\*\* for their valuable guidance and support throughout this project. And this project is Implemented from the dataset and method provided in \[SignFi](https://yongsen.github.io/SignFi/) Paper.







\###### \*\*Author\*\*







\*\*Joy Saha\*\*



Bangladesh University of Engineering and Technology (BUET)











\###### \*\*References\*\*







Y. Ma, G. Zhou, S. Wang, H. Zhao, and W. Jung, “Signfi: Sign language recog



nition using wifi,” Proceedings of the ACM on Interactive, Mobile, Wearable 



and Ubiquitous Technologies, vol. 2, no. 1, pp. 1–21, 2018.









\#### \*\*License\*\*















This project is for academic and educational purposes.





