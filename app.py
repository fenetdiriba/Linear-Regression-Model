import numpy as np
import streamlit as st
from LinearRegression import (
    LinearRegression,
)  # Import your custom class from your repository

st.title("Linear Regression Model from Scratch")
st.write(
    "This app demonstrates gradient descent and MSE optimization built from"
    " scratch."
)

# Example UI inputs
st.sidebar.header("Model Hyperparameters")
learning_rate = st.sidebar.slider(
    "Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, step=0.0001
)
iterations = st.sidebar.slider(
    "Iterations", min_value=100, max_value=5000, value=1000, step=100
)

# Generate sample data for demonstration
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

st.subheader("Generated Dataset Preview")
st.write(f"Dataset shape: X={X.shape}, y={y.shape}")

# Train button
if st.button("Train Model"):
  # Initialize and fit your custom model
  model = LinearRegression(lr=learning_rate, n_iters=iterations)
  model.fit(X, y)

  st.success("Model trained successfully!")
  st.write(f"Optimized Weights: {model.weights}")
  st.write(f"Optimized Bias: {model.bias}")
