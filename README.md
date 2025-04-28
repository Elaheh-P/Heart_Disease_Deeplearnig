# Introduction

Heart disease is one of the leading causes of mortality worldwide, and early detection plays a crucial role in improving patient outcomes. This project focuses on developing a deep learning model to predict the presence of heart disease based on various clinical features. The model is built using a structured dataset and employs thorough preprocessing techniques to optimize performance. To ensure a reliable and unbiased evaluation, stratified K-fold cross-validation is utilized, allowing for consistent assessment across different subsets of the data. The objective is to demonstrate the potential of deep learning methods in supporting clinical decision-making and enhancing the early diagnosis of heart conditions.

## Table of Contents

- [Dataset](https://www.notion.so/Heart-Disease-Prediction-using-Deep-Learning-1e3045922c458025a13ec2622329ef9d?pvs=21)
- [Project Structure](https://www.notion.so/Heart-Disease-Prediction-using-Deep-Learning-1e3045922c458025a13ec2622329ef9d?pvs=21)
- [Setup and Installation](https://www.notion.so/Heart-Disease-Prediction-using-Deep-Learning-1e3045922c458025a13ec2622329ef9d?pvs=21)
- [Model Architecture](https://www.notion.so/Heart-Disease-Prediction-using-Deep-Learning-1e3045922c458025a13ec2622329ef9d?pvs=21)
- [Training Process](https://www.notion.so/Heart-Disease-Prediction-using-Deep-Learning-1e3045922c458025a13ec2622329ef9d?pvs=21)
- [Results](https://www.notion.so/Heart-Disease-Prediction-using-Deep-Learning-1e3045922c458025a13ec2622329ef9d?pvs=21)
- [Visualization](https://www.notion.so/Heart-Disease-Prediction-using-Deep-Learning-1e3045922c458025a13ec2622329ef9d?pvs=21)
- [Acknowledgements](https://www.notion.so/Heart-Disease-Prediction-using-Deep-Learning-1e3045922c458025a13ec2622329ef9d?pvs=21)

## Dataset

The dataset used (`heart.csv`) is loaded from Google Drive and contains information about various patient features such as:

| Feature | Description |
| --- | --- |
| **Age** | The age of the patient (in years). |
| **Sex** | The gender of the patient: 1 = Male, 0 = Female. |
| **ChestPainType** | Type of chest pain experienced: - `TA`: Typical Angina - `ATA`: Atypical Angina- `NAP`: Non-A- `ASY`ASY`: Asymptomatic |
| **RestingBP** | Resting blood pressure (in mm Hg) when the patient was admitted. |
| **Cholesterol** | Serum cholesterol level (in mg/dl). |
| **FastingBS** | Fasting blood sugar > 120 mg/dl: - 1 = True - 0 = False |
| **RestingECG** | Resting electrocardiogram results: - `Normal` - `ST` (having ST-T wave abnormality) - `LVH` (probable or definite left ventricular hypertrophy) |
| **MaxHR** | Maximum heart rate achieved during exercise. |
| **Exercise** | Whether the patient experienced exercise-induced angina (chest pain): - `Y` = Yes - `N` = No |
| **Old peak** | Depression of the ST segment induced by exercise relative to rest (a measure of abnormal heart function). |
| **ST_Slope** | The slope of the peak exercise ST segment: - `Up` - `Flat` - `Down` |
| **HeartDisease** | Target variable: - 1 = Presence of heart disease - 0 = No heart disease |

**Target Variable**: `HeartDisease` (Binary classification: 0 = No Heart Disease, 1 = Presence of Heart Disease)

## Project Structure

- **Data Loading**: Loads heart disease dataset from Google Drive.
- **Exploratory Data Analysis (EDA)**: Visualizes class distributions and feature correlations.
- **Preprocessing**: Encodes categorical features and normalizes the data.
- **Modeling**: A deep learning model is built using TensorFlow/Keras.
- **Cross-Validation**: Uses 5-fold Stratified K-Fold cross-validation for evaluation.
- **Visualization**: Accuracy and loss plots over training epochs.

## Setup and Installation

1. **Environment**: The code is designed to run in Google Colab.
2. **Libraries**:
    - TensorFlow
    - Pandas
    - NumPy
    - Scikit-learn
    - Seaborn
    - Matplotlib

## Model Architecture

The deep learning model follows a simple feedforward neural network structure:

- **Input Layer**: 11 features
- **Hidden Layer 1**: 64 neurons, ReLU activation, 30% Dropout
- **Hidden Layer 2** : 32 neurons, ReLU activation
- **Output Layer**: 1 neuron, Sigmoid activation (for binary classification)

Optimizer: **Adam**

Loss Function: **Binary Crossentropy**

Metric: **Accuracy**

## Training Process

- **Cross-Validation**: 5-Fold Stratified
- **Early Stopping**: Monitors validation loss to prevent overfitting
- **Batch Size**: 128
- **Epochs**: Up to 50 (with early stopping if necessary)

Each fold trains a separate model and evaluates it on a validation split, reporting accuracy and loss per fold.

## Results

The model's performance was evaluated using 5-fold stratified cross-validation.

### 📋 Fold-Wise Performance

| Fold | Validation Accuracy | Validation Loss |
| --- | --- | --- |
| 1 |  87.50% | 0.3281  |
| 2 | 82.61% | 0.3570 |
| 3 | 81.52% | 0.4209  |
| 4 | 89.07% |  0.3060 |
| 5 | 84.15% | 0.3592 |

### 📈 Average Performance

- **Average Validation Accuracy**:  84.97%
- **Average Validation Loss**:  0.3543

### 📊 Interpretation

- The model demonstrates **strong and consistent accuracy** across all folds, with low validation loss.
- **No major overfitting** was observed: training and validation metrics improved together without divergence.
- **Early stopping** effectively prevented overtraining.

---

## Visualization

### 📊 1. Class Distribution Plot

**Description**) heart disease:
![Image](https://github.com/user-attachments/assets/6d70aaee-61e3-4429-9aca-c709ad1e4902)

A bar plot showing the count of patients with (`1`) and without (`0`) heart disease.

- The classes are relatively **balanced**, with a slightly higher number of patients with heart disease.
- This balance is beneficial because it means the model won't be biased towards predicting one class over another during training.

---

### 🧩 2. Feature Correlation Heatmap

**Description**:
![Image](https://github.com/user-attachments/assets/0cc541e1-0318-4eac-ad4d-c57312c215b4)

A heatmap illustrating Pearson correlation coefficients between all encoded features and the target variable (`HeartDisease`).

ExerciseAngina:
Exhibits a strong positive correlation with heart disease. Patients who experience angina (chest pain) during exercise are much more likely to be diagnosed with heart disease, making this feature a critical clinical indicator of cardiac risk.

Sex:
Shows a positive correlation, indicating that males (Sex = 1) are at a significantly higher risk of heart disease compared to females. This aligns with known epidemiological trends where heart disease prevalence is often greater among male populations.

Oldpeak:
Displays a moderate positive correlation with heart disease. Higher Oldpeak values — reflecting greater levels of ST segment depression during exercise — are typically associated with more severe cardiac abnormalities, thereby increasing the likelihood of heart disease diagnosis.

Overall:
These features capture key physiological and demographic risk factors, and are expected to be among the most influential contributors to the model’s predictive capability. Their strong associations with heart disease underline their importance in both clinical evaluation and automated prediction systems.



---

### 📈 3. Individual Feature Distributions

**Description**:

Histograms with Kernel Density Estimation (KDE) plotted for `ExerciseAngina`, `Sex`, and `Oldpeak`.

**Insights (Based on Results)**:

- **ExerciseAngina**:
    - Majority of patients do **not** experience angina (`ExerciseAngina = 0`), but those who do are more likely to have heart disease based on correlation.
- **Sex**:
    - There are **more males than females** in the dataset.
    - Since males are more correlated with heart disease, the slight imbalance might influence predictions toward higher risk for males.
- **Oldpeak** :
    - Most patients have a **low Oldpeak value**, but a smaller group has significantly higher values.
    - Higher Oldpeak values (more severe ST depression) are associated with an increased likelihood of heart disease.

Understanding these distributions helped in feature engineering and gave context for the model's learning.

---

## 📈 4. Training and Validation Accuracy Plot

The training and validation accuracy curves showed a steady increase over epochs, demonstrating that the model was able to learn meaningful patterns from the data without overfitting. Throughout training, there was no significant divergence between the training and validation accuracy lines — both curves rose together in a consistent manner. This close alignment suggests that the model was generalizing well to unseen validation data, rather than simply memorizing the training set. Additionally, early stopping was employed effectively: training halted at the point where validation performance no longer improved, ensuring the final model retained the best weights without undergoing unnecessary extra epochs.

---

## 📉 5. Training and Validation Loss Plot

The training and validation loss curves both decreased together during the initial training epochs, indicating that the model was successfully minimizing the loss function on both training and validation data. After several epochs, the curves began to flatten, suggesting that the model had converged and reached a point of stable performance. Importantly, there was no evidence of severe overfitting: the validation loss did not begin to increase while training loss continued to drop, which would have indicated model over-specialization. Instead, both loss metrics behaved consistently, supporting the conclusion that the model trained efficiently and retained good generalization capability to new data.
