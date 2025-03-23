import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sd, option):
    d1 = (np.log(S / K) + (r + 0.5 * sd**2) * T) / (sd * np.sqrt(T))
    d2 = d1 - sd * np.sqrt(T)

    if option.lower() == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def plot_option_price_heatmap(K, T, r, option):
    S_range = np.linspace(50, 150, 50)
    sigma_range = np.linspace(0.1, 1.0, 50)
    prices = np.zeros((len(sigma_range), len(S_range)))

    for i, sigma in enumerate(sigma_range):
        for j, S_val in enumerate(S_range):
            prices[i, j] = black_scholes_price(S_val, K, T, r, sigma, option)

    plt.figure(figsize=(4, 2.5))
    ax = sns.heatmap(
        prices,
        xticklabels=10,
        yticklabels=5,
        cmap="viridis",
        cbar_kws={'label': 'Option Price'}
    )
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)

    colorbar = ax.collections[0].colorbar
    colorbar.set_label("Option Price", fontsize=5, labelpad=25)
    colorbar.ax.tick_params(labelsize=7)

    plt.title(f"Option Price Heatmap ({option.capitalize()} Option)", fontsize=5)
    plt.xlabel("Stock Price (S)", fontsize=5)
    plt.ylabel("Volatility (σ)", fontsize=5)
    plt.tight_layout()
    st.pyplot(plt)


st.set_page_config(layout="wide")

st.title("Black-Scholes Option Pricing Calculator")

col1, col2 = st.columns([1, 2])

with col1:
    S = st.number_input("Current Stock Price (S)", value=100.0)
    K = st.number_input("Strike Price (K)", value=150.0)
    T = st.number_input("Time to Expiry (T in years)", value=1.0)
    r = st.number_input("Risk-Free Interest Rate (%)", value=5.0)
    sd = st.number_input("Volatility (σ in %)", value=50.0)
    option = st.radio("Option Type", ("call", "put"))
    button = st.button("Calculate")
    if button:
        price = black_scholes_price(S, K, T, r / 100, sd / 100, option)
        st.markdown(f"### 💰 Option Price: **${price:.2f}**")   

with col2:
    if button:
        st.markdown("### 📊 Option Price Heatmap")
        with st.spinner("Generating heatmap..."):
            plot_option_price_heatmap(K, T, r / 100, option)
