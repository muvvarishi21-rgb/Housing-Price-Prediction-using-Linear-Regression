# 🏠 Housing Price Prediction using Linear Regression

## 📌 Overview
This project implements a Machine Learning regression model to predict housing prices using the **Linear Regression algorithm**.  
It covers the complete ML workflow including data preprocessing, exploratory data analysis (EDA), model training, and performance evaluation.

---

## 🎯 Objective
To build a predictive model that estimates housing prices based on multiple input features and evaluates performance using standard regression metrics.

---

## 📊 Dataset
- File: `train.csv`
- Target Variable: `MEDV` (Median value of owner-occupied homes)
- Features: Various housing-related attributes

The dataset is analyzed for:
- Missing values
- Feature correlations
- Data distribution patterns

---

## 🛠 Tech Stack

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 🔄 Project Workflow

### 1️⃣ Data Loading & Inspection
- Loaded dataset using Pandas
- Checked structure, shape, and missing values

### 2️⃣ Exploratory Data Analysis (EDA)
- Generated correlation matrix
- Visualized correlations using heatmap
- Identified important predictors

### 3️⃣ Data Preprocessing
- Train-Test Split (80:20)
- Feature Scaling using StandardScaler

### 4️⃣ Model Development
- Applied Linear Regression
- Trained model on training dataset

### 5️⃣ Model Evaluation
Performance evaluated using:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score (Coefficient of Determination)**

Also visualized:
- Actual vs Predicted values

---

## 📈 Results
The model demonstrates strong predictive capability with optimized regression performance metrics.

---

## 🚀 Installation & Execution

### Clone the Repository
```bash
git clone https://github.com/your-username/housing-price-prediction.git
cd housing-price-prediction
```

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Project
```bash
jupyter notebook
```

Open the notebook file and execute all cells.

---

## 📂 Project Structure
```
housing-price-prediction/
│
├── train.csv
├── Housing_Price_Prediction.ipynb
└── README.md
```

---

## 🔮 Future Enhancements
- Implement advanced models (Random Forest, XGBoost)
- Perform cross-validation
- Hyperparameter tuning
- Deploy as a web application

---

## 👨‍💻 Author
**Your Name**  
Machine Learning Enthusiast  

---

## 📜 License
This project is for educational purposes.
