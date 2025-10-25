# Diamond-Dynamics-Price-Prediction-and-Market-Segmentation
An end-to-end Machine Learning project that predicts diamond prices and segments diamonds into market categories using Regression and Clustering algorithms. Includes advanced EDA, feature engineering, and a Streamlit-based interactive UI for real-time prediction and segmentation. Tech Stack: Python, Scikit-learn, TensorFlow, K-Means, PCA, Streamlit


# 💎 Diamond Dynamics: Price Prediction and Market Segmentation

## 📌 Overview
**Diamond Dynamics** is an end-to-end Machine Learning and Data Analytics project that predicts diamond prices and segments diamonds into market groups based on their physical and qualitative attributes.  
It integrates **price prediction using regression models** and **market segmentation using clustering** — deployed through an interactive **Streamlit UI**.

---

## 🎯 Objectives
- Predict diamond prices using ML models (Linear Regression, Random Forest, XGBoost, ANN).
- Segment diamonds into clusters using K-Means for market insights.
- Design an intuitive Streamlit UI for user-driven price and segment predictions.

---

## 🧠 Skills Applied
- Data Cleaning & Preprocessing  
- EDA & Data Visualization  
- Feature Engineering & Selection  
- Outlier and Skewness Handling  
- Regression (ML + ANN)  
- K-Means Clustering  
- Dimensionality Reduction (PCA)  
- Streamlit Web App Development  

---

## 🛍️ Domain
**E-Commerce | Luxury Goods Analytics | Retail Pricing Optimization**

---

## 💡 Real-World Use Cases
- Dynamic pricing for diamond retailers  
- Market segmentation for inventory optimization  
- Customer segmentation for personalized marketing  
- Luxury product recommendation systems  

---

## 📊 Dataset
**Source:** Diamond dataset  
**Shape:** 53,940 rows × 10 features  

| Column | Description |
|--------|--------------|
| carat | Weight of the diamond |
| cut | Quality of the cut (Fair, Good, Very Good, Premium, Ideal) |
| color | Color grading (D–J) |
| clarity | Measure of inclusions (IF, VVS1, VVS2, etc.) |
| depth, table | Proportion measurements |
| x, y, z | Dimensions (mm) |
| price | Price in USD (converted to INR) |

---

## ⚙️ Project Workflow
### 1️⃣ **Data Preprocessing**
- Handle missing or invalid data (x, y, z = 0)
- Treat outliers (IQR / Z-Score)
- Fix skewness using log/sqrt transformation

### 2️⃣ **Exploratory Data Analysis**
- Distribution & correlation analysis  
- Boxplots and pairplots for price vs carat, cut, clarity  
- Bar charts for average price per category  

### 3️⃣ **Feature Engineering**
Derived new features such as:
- `Volume = x * y * z`
- `Price per Carat = price / carat`
- `Dimension Ratio = (x + y) / (2 * z)`
- `Carat Category` (Light, Medium, Heavy)

### 4️⃣ **Model Building**
- **Regression Models:** Linear Regression, Decision Tree, Random Forest, XGBoost, ANN  
- **Clustering Models:** K-Means, DBSCAN, Hierarchical Clustering  
- **Model Evaluation:** MAE, MSE, RMSE, R², Silhouette Score  

### 5️⃣ **Deployment**
Developed an interactive **Streamlit Web App** for:
- Price Prediction (Regression)
- Market Segmentation (Clustering)
- Real-time visualization and output

---

## 💻 Streamlit Features
- User inputs: carat, cut, color, clarity, dimensions (x, y, z)
- Predicts **Price (in INR)** and **Market Cluster**
- Displays **cluster name and segment insights**
- Clean and interactive UI

---

## 🧩 Technologies Used
- Python  
- Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn, TensorFlow/Keras  
- Streamlit  
- Pickle for model serialization  

---

## 🧠 Cluster Naming
| Cluster | Description |
|----------|-------------|
| Premium Heavy Diamonds | Large, expensive, premium-grade stones |
| Affordable Small Diamonds | Small, budget-friendly stones |
| Mid-range Balanced Diamonds | Moderate price and size balance |

---

## 📦 Deliverables
- Jupyter Notebook with all ML workflows  
- Trained Regression & Clustering models (`.pkl` files)  
- Streamlit App (`app.py`)  
- Visualizations and model evaluation reports  

---

## 🏁 Conclusion
This project demonstrates how **data-driven insights and ML models** can optimize **luxury pricing** and **market segmentation**.  
It bridges **E-commerce analytics** with **machine learning applications** for smarter business decisions.

---

## 🔗 Connect
**Developed by:** Muthu Selvam  
💼 [LinkedIn](https://www.linkedin.com/in/muthu-selvam-359318250)  
📧 muthuselvam25042003@gmail.com

