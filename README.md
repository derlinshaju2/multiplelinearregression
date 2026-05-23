# 🏘️ Advanced Home Price Predictor
> A Multivariate Linear Regression model built with **Streamlit** and **Scikit-Learn** to predict real estate value using multiple property features.

---

## 🔗 Live Demo
Predict house prices in real-time here:  
** https://multiplelinearregression-zqhgc5ej6nbptrncycej54.streamlit.app/ **

---

## ✨ Features
Unlike simple regression, this model considers three distinct variables to determine market value:

* **📐 Square Footage (Area):** The primary driver of property value.
* **🛏️ Bedroom Count:** Evaluates how the number of rooms affects the final price.
* **⏳ Property Age:** Accounts for depreciation or historical value over time.
* **🧹 Automated Data Cleaning:** Uses Pandas to handle missing values by calculating and filling the mean for the `bedrooms` column.

---

## 🛠️ Technical Implementation
This project follows a complete ML pipeline:

1.  **Data Acquisition:** Loads raw data directly from a remote GitHub CSV source.
2.  **Preprocessing:** Handles `NaN` (null) values using the `.fillna()` method based on statistical averages.
3.  **Modeling:** Implements `sklearn.linear_model.LinearRegression` to solve the equation:
    $$Price = (m_1 \cdot \text{Area}) + (m_2 \cdot \text{Bedrooms}) + (m_3 \cdot \text{Age}) + b$$
4.  **UI/UX:** Built with Streamlit for a clean, interactive user experience.

---

├── app.py                # Streamlit application & ML logic
├── requirements.txt      # Dependency list (Pandas, Sklearn, etc.)
└── README.md             # Project documentation
