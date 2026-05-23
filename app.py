import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="Advanced Home Price Predictor", layout="centered")

st.title("🏘️ Advanced Home Price Predictor")
st.write("This model uses **Multivariate Linear Regression** to predict prices based on Area, Bedrooms, and Age.")

# 1. Load and Clean Data
@st.cache_data # Caches data so it doesn't reload on every click
def load_data():
    url = 'https://raw.githubusercontent.com/codebasics/py/refs/heads/master/ML/2_linear_reg_multivariate/homeprices.csv'
    df = pd.read_csv(url)
    # Filling NaN in bedrooms with the rounded mean
    bed_mean = df['bedrooms'].mean()
    df['bedrooms'] = df['bedrooms'].fillna(round(bed_mean))
    return df

df = load_data()

# 2. Train the Model
X = df[['area', 'bedrooms', 'age']]
y = df['price']
model = LinearRegression()
model.fit(X, y)

# --- SIDEBAR INPUTS ---
st.sidebar.header("Property Specifications")
input_area = st.sidebar.number_input("Total Area (sq ft):", min_value=1000, max_value=10000, value=3000)
input_beds = st.sidebar.slider("Number of Bedrooms:", 1, 10, 3)
input_age = st.sidebar.number_input("Age of the House (Years):", min_value=0, max_value=100, value=15)

# 3. Prediction Logic
if st.sidebar.button("Predict Price"):
    prediction = model.predict([[input_area, input_beds, input_age]])
    
    # Display Results
    st.subheader("Price Prediction Results")
    st.metric(label="Estimated Market Value", value=f"${prediction[0]:,.2f}")
    
    # Show the formula impact
    st.write(f"### How it was calculated:")
    st.write(f"- **Area impact:** ${model.coef_[0]:,.2f} per sq ft")
    st.write(f"- **Bedroom impact:** ${model.coef_[1]:,.2f} per room")
    st.write(f"- **Age impact:** ${model.coef_[2]:,.2f} per year (depreciation)")
else:
    st.info("Adjust the settings in the sidebar and click **Predict Price**.")

# 4. Data Preview
with st.expander("View Training Dataset"):
    st.dataframe(df)
