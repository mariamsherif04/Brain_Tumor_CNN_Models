# Brain Tumor Classification Using CNNs and Transfer Learning

## Overview
This project focuses on classifying brain MRI images into four categories: **glioma**, **meningioma**, **notumor**, and **pituitary**. We explore multiple convolutional neural network (CNN) architectures, regularization techniques, and transfer learning models (MobileNetV2 and ResNet50) to achieve optimal performance. 
---

## Dataset
The dataset consists of labeled brain MRI images categorized into four classes:
- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

**Key details:**
- Images are resized to 224×224 pixels for CNN input.
- Dataset split: Training, Validation, and Test sets.

---

## Data Preparation
Preprocessing steps applied include:
1. **Resizing** to 224×224.
2. **Normalization** (scaling pixel values to [0,1]).
3. **Data augmentation**: rotations, flips, zooms, and shifts to increase diversity.
4. Conversion to TensorFlow datasets for efficient batch processing.

---

## Model Experiments
We trained 20 different models in phases:

### Phase 1 – Preprocessing (Baseline)
1. **Model 1**: Custom CNN, resizing only  
2. **Model 2**: Custom CNN, resizing + normalization  
3. **Model 3**: Custom CNN, resizing + normalization + augmentation  

### Phase 2 – Architecture Variations
4. **Model 4**: Shallow CNN (2 conv layers, 16→32 filters)  
5. **Model 5**: Medium CNN (3–4 conv layers, 32→128 filters)  
6. **Model 6**: Deep CNN (5+ conv layers, 64→256 filters)  
7. **Model 7**: CNN with 5×5 filters  
8. **Model 8**: CNN with LeakyReLU activation  

### Phase 3 – Optimizers & Hyperparameters
9. **Model 9**: CNN + SGD (lr=0.01, momentum=0.9)  
10. **Model 10**: CNN + RMSprop (lr=1e-4)  
11. **Model 11**: CNN + Adam with different batch sizes  
12. **Model 12**: CNN + Adam + varied dropout (0.2 vs 0.5)  

### Phase 4 – Regularization Techniques
13. **Model 13**: CNN + Dropout (0.3)  
14. **Model 14**: CNN + Batch Normalization  
15. **Model 15**: CNN + L2 weight regularization  
16. **Model 16**: CNN + Dropout + Batch Normalization  

### Phase 5 – Transfer Learning
17. **Model 17**: MobileNetV2 (frozen base)  
18. **Model 18**: MobileNetV2 (fine-tune last 20 layers)  
19. **Model 19**: ResNet50 (frozen base)  
20. **Model 20**: ResNet50 (fine-tune last 20 layers)  

---

## Model Architectures
- Custom CNNs consist of 2–5 convolutional layers, followed by max pooling, dropout/batch normalization, and fully connected layers.  
- Transfer learning models use pretrained ImageNet weights, with either frozen or fine-tuned last layers.

---

## Training Details
- Optimizers: **Adam**, **SGD**, **RMSprop**  
- Loss: Sparse categorical cross-entropy  
- Metrics: Accuracy  
- Training performed on **CPU**.
- Early stopping applied on validation loss to prevent overfitting  

---


### Summary of Model Performance
`Best to worst`
| Model | Name & Description | Test Accuracy | Test Loss | Val Accuracy | Val Loss | Training Time | Key Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 18** | **MobileNetV2 (fine-tuned)** | **0.9672** | **0.1140** | 0.9648 | 0.1992 | 16 min | **Best overall performance.** Transfer learning with fine-tuning works best. |
| **Model 11** | Medium CNN + LeakyReLU + Adam (bs=16) | 0.9573 | 0.1746 | 0.9653 | 0.1417 | - | Best custom CNN. Optimal batch size with Adam. |
| **Model 13** | Model 11 + Dropout (0.3) | 0.9565 | 0.1928 | 0.9560 | 0.2037 | 7 min | Good regularization, slightly lower performance than Model 11. |
| **Model 8** | Medium CNN + Leaky ReLU | 0.9542 | 0.1816 | 0.9625 | 0.2051 | 7 min | Strong baseline. Leaky ReLU improves on standard ReLU. |
| **Model 14** | Model 11 + BatchNorm | 0.9535 | 0.2145 | 0.9533 | 0.2052 | 13 min | Stable training but no significant gain over Model 11. |
| **Model 7** | Medium CNN + 5×5 filters | 0.9527 | 0.1865 | 0.9643 | 0.2062 | 10 min | Larger filters captured features effectively. |
| **Model 15** | Model 11 + L2 Regularization | 0.9527 | 0.2585 | 0.9560 | 0.2545 | 11 min | Higher loss suggests regularization may be too strong. |
| **Model 2** | CNN + Resize + Normalization | 0.9169 | 0.2335 | 0.9527 | 0.2391 | - | Good baseline. Normalization improved over Model 1. |
| **Model 5** | Medium CNN (ReLU) | 0.9169 | 0.2430 | 0.9589 | 0.2135 | 4 min | Deeper architecture matched simpler model's performance. |
| **Model 4** | Shallow CNN | 0.9161 | 0.2211 | 0.9420 | 0.2205 | 1 min | Surprisingly effective for a very simple model. |
| **Model 1** | Baseline CNN + Resize only | 0.9054 | 0.3493 | 0.9464 | 0.2723 | - | Basic model, higher loss without normalization. |
| **Model 10** | Medium CNN + LeakyReLU + RMSprop | 0.9451 | 0.2092 | 0.9509 | 0.2808 | - | Good, but Adam (Model 11) was better. |
| **Model 9** | Medium CNN + LeakyReLU + SGD | 0.9275 | 0.2300 | 0.9527 | 0.2339 | 5 min | SGD with momentum performed worse than Adam. |
| **Model 6** | Simplified Deep CNN | 0.9443 | 0.2490 | 0.9509 | 0.2843 | 32 min | More complex but outperformed by simpler models. |
| **Model 16** | Model 11 + Dropout + BatchNorm | 0.9443 | 0.1566 | 0.9542 | 0.1714 | 18 min | Lowest test loss, but accuracy slightly reduced. |
| **Model 17** | MobileNetV2 (frozen) | 0.9428 | 0.1766 | 0.9428 | 0.2102 | 19 min | Strong performance without any fine-tuning. |
| **Model 12** | Model 8 + Dropout (0.5) | 0.9550 | 0.1863 | 0.9571 | 0.1857 | 8 min | Good regularization with dropout. |
| **Model 20** | ResNet50 (fine-tuned) | 0.8963 | 0.3093 | 0.9181 | 0.3023 | 60 min | Underperformed compared to MobileNetV2. |
| **Model 3** | Light Augmentation + Norm | 0.7864 | 0.6531 | 0.8804 | 0.5312 | - | Augmentation hurt performance, likely too aggressive. |
| **Model 19** | ResNet50 (frozen) | 0.6110 | 0.9856 | 0.6620 | 0.8836 | 83 min | Poor feature extraction for this dataset. |

---

### Key Conclusions

1.  **Top Performers:** The fine-tuned **MobileNetV2 (Model 18)** was the clear winner, followed closely by the well-tuned custom **Medium CNN with Leaky ReLU and Adam (Model 11)**.
2.  **Effective Preprocessing:** Simply adding **normalization (Model 2)** provided a significant boost over the baseline (Model 1).
3.  **Architecture Matters:** The **Medium CNN (Model 5)** was a sweet spot. Using **5x5 filters (Model 7)** and **Leaky ReLU (Model 8)** provided further improvements.
4.  **Optimizer Choice:** **Adam** was the most effective optimizer for this task, outperforming RMSprop and SGD.
5.  **Regularization Trade-off:** While Dropout and BatchNorm helped prevent overfitting (evident in lower gaps between train/val loss), they sometimes slightly reduced the peak test accuracy compared to the best unregularized model (Model 11).
6.  **Transfer Learning:** **MobileNetV2** was exceptionally well-suited for this task, even when frozen. **ResNet50** performed poorly, suggesting its features were not a good fit for this specific dataset.


