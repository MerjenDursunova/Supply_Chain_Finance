import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


st.set_page_config(
    page_title="Supply Chain Finance Advisor",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent / 'outputs' / 'demand_forecast_model.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()


def prepare_features(df):
    df = df.copy()
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    df['lag_1'] = df['demand'].shift(1)
    df['lag_7'] = df['demand'].shift(7)
    df['lag_30'] = df['demand'].shift(30)
    df['rolling_mean_7'] = df['demand'].shift(1).rolling(7).mean()
    df['rolling_mean_30'] = df['demand'].shift(1).rolling(30).mean()
    return df.dropna()

def classify_risk(cv):
    if cv < 0.10:
        return 'Low Risk'
    elif cv < 0.20:
        return 'Medium Risk'
    return 'High Risk'

def get_recommendation(production_cost, retail_price,
                      risk_level, risk_adjusted_demand):
    cost_ratio = production_cost / retail_price
    risk_modifier = {
        'Low Risk': 0,
        'Medium Risk': 0.05,
        'High Risk': 0.10
    }[risk_level]
    
    low_threshold = 0.40 + risk_modifier
    high_threshold = 0.70 + risk_modifier

    if cost_ratio <= low_threshold:
        method = 'Early Payment'
        color = 'green'
        rationale = (
            f"Production cost is low relative to retail price "
            f"(cost ratio: {cost_ratio:.2f}). The benefit from "
            f"increased production outweighs any interest income. "
            f"Retailer should prepay the manufacturer."
        )
    elif cost_ratio <= high_threshold:
        method = 'In-House Factoring'
        color = 'orange'
        rationale = (
            f"Production cost is moderate (cost ratio: {cost_ratio:.2f}). "
            f"Retailer can earn interest income while maintaining "
            f"reasonable production levels via a financing subsidiary."
        )
    else:
        method = 'Bank Financing'
        color = 'red'
        rationale = (
            f"Production cost is high (cost ratio: {cost_ratio:.2f}). "
            f"Financial risk is too high for the retailer to absorb. "
            f"Manufacturer should seek external bank financing."
        )
    
    return method, color, rationale, cost_ratio


st.title("Supply Chain Finance Advisor")
st.markdown(
    "Based on **Chen, Lu & Cai (2020)** — *Buyer Financing in "
    "Pull Supply Chains*. This system predicts demand, estimates "
    "risk, and recommends the optimal financing method."
)

st.divider()

st.sidebar.header("Supplier Parameters")

retail_price = st.sidebar.number_input(
    "Retail Price ($ per unit)",
    min_value=1.0, max_value=1000.0,
    value=20.0, step=0.5
)

production_cost = st.sidebar.number_input(
    "Production Cost ($ per unit)",
    min_value=0.1, max_value=retail_price,
    value=8.0, step=0.5
)

st.sidebar.divider()
st.sidebar.header("Upload Sales Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with 'date' and 'demand' columns",
    type=['csv']
)


if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file, parse_dates=['date'])
    df = prepare_features(raw_df)

    features = [
        'day_of_year', 'month', 'quarter', 'year',
        'day_of_week', 'lag_1', 'lag_7', 'lag_30',
        'rolling_mean_7', 'rolling_mean_30'
    ]

    df['predicted_demand'] = model.predict(df[features])

    mean_demand = df['demand'].mean()
    std_demand = df['demand'].std()
    cv = std_demand / mean_demand
    risk_level = classify_risk(cv)
    bias = (df['demand'] - df['predicted_demand']).mean()
    var_95 = np.percentile(
        (df['demand'] - df['predicted_demand']).abs(), 95
    )

    risk_adjusted_demand = {
        'Low Risk': mean_demand + 0.5 * std_demand,
        'Medium Risk': mean_demand,
        'High Risk': mean_demand - 0.5 * std_demand
    }[risk_level]

    method, color, rationale, cost_ratio = get_recommendation(
        production_cost, retail_price,
        risk_level, risk_adjusted_demand
    )

    st.header("Block 1: Demand Forecast")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['date'], df['demand'],
            label='Actual Demand', color='steelblue', alpha=0.7)
    ax.plot(df['date'], df['predicted_demand'],
            label='Predicted Demand', color='red',
            alpha=0.7, linestyle='--')
    ax.set_title('Actual vs Predicted Demand')
    ax.set_xlabel('Date')
    ax.set_ylabel('Units')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.header("Block 2: Demand Risk Estimation")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Demand", f"{mean_demand:.0f} units")
    col2.metric("Volatility (CV)", f"{cv*100:.1f}%")
    col3.metric("Risk Level", risk_level)
    col4.metric("VaR 95%", f"±{var_95:.0f} units")

    st.header("Block 3: Financing Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Production Cost", f"${production_cost}/unit")
        st.metric("Retail Price", f"${retail_price}/unit")
        st.metric("Cost Ratio", f"{cost_ratio:.2f}")

    with col2:
        st.markdown("### Recommended Method:")
        if color == 'green':
            st.success(f" {method}")
        elif color == 'orange':
            st.warning(f" {method}")
        else:
            st.error(f" {method}")

        st.markdown("**Justification:**")
        st.info(rationale)

    st.divider()
    st.subheader(" Download Results")
    def generate_html_report(retail_price, production_cost, cost_ratio,
                             mean_demand, cv, risk_level, var_95,
                             method, rationale, bias):
        
        color_map = {
            'Early Payment': '#9B7CB6',      
            'In-House Factoring': '#F0A868', 
            'Bank Financing': '#E07A7A' 
        }
        method_color = color_map[method]
        
        html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
                max-width: 800px;
                margin: 40px auto;
                color: #333;
                line-height: 1.6;
                background: #f4f5f7;
            }}
            .header {{
                background: #3B2F4D;
                color: white;
                padding: 32px;
                border-radius: 6px;
                margin-bottom: 30px;
                box-shadow: 0 4px 12px rgba(59, 47, 77, 0.15);
            }}
            .header h1 {{
                margin: 0 0 8px 0;
                font-size: 24px;
                font-weight: 600;
                letter-spacing: -0.5px;
            }}
            .header p {{
                margin: 0;
                opacity: 0.85;
                font-size: 14px;
            }}
            .section {{
                background: #ffffff;
                border-radius: 6px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }}
            .section h2 {{
                color: #3B2F4D;
                margin-top: 0;
                font-size: 16px;
                font-weight: 600;
                border-bottom: 2px solid #6A4C8C;
                padding-bottom: 10px;
                margin-bottom: 16px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
            }}
            .metric-box {{
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 14px;
            }}
            .metric-label {{
                font-size: 12px;
                color: #6c757d;
                margin-bottom: 6px;
                font-weight: 500;
            }}
            .metric-value {{
                font-size: 18px;
                font-weight: 700;
                color: #3B2F4D;
            }}
            .recommendation-box {{
                background: {method_color};
                color: white;
                border-radius: 6px;
                padding: 20px;
                margin-bottom: 16px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            }}
            .recommendation-box h2 {{
                margin: 0 0 8px 0;
                font-size: 22px;
                font-weight: 600;
            }}
            .recommendation-box p {{
                margin: 0;
                opacity: 0.9;
                font-size: 14px;
            }}
            .justification {{
                background: white;
                border-left: 4px solid {method_color};
                padding: 16px;
                border-radius: 0 4px 4px 0;
                font-size: 14px;
                color: #444;
            }}
            .footer {{
                text-align: center;
                font-size: 12px;
                color: #999;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #dee2e6;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 12px;
            }}
            th {{
                background: #5C456E;
                color: white;
                padding: 12px;
                text-align: left;
                font-size: 13px;
                font-weight: 600;
            }}
            td {{
                padding: 12px;
                border-bottom: 1px solid #e9ecef;
                font-size: 13px;
            }}
            tr:nth-child(even) td {{
                background: #f8f9fa;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1> Supply Chain Finance Recommendation Report</h1>
            <p>Generated: {pd.Timestamp.now().strftime('%B %d, %Y at %H:%M')} &nbsp;|&nbsp; 
               Based on Chen, Lu & Cai (2020)</p>
        </div>

        <div class="section">
            <h2>Supplier Parameters</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">Retail Price</div>
                    <div class="metric-value">${retail_price}/unit</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Production Cost</div>
                    <div class="metric-value">${production_cost}/unit</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Cost Ratio</div>
                    <div class="metric-value">{cost_ratio:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Profit Margin</div>
                    <div class="metric-value">
                        {((retail_price - production_cost)/retail_price*100):.1f}%
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Demand Risk Analysis</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">Mean Demand</div>
                    <div class="metric-value">{mean_demand:.0f} units</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Demand Volatility (CV)</div>
                    <div class="metric-value">{cv*100:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value">{risk_level}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Worst Case Error (VaR 95%)</div>
                    <div class="metric-value">±{var_95:.0f} units</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Financing Recommendation</h2>
            <div class="recommendation-box">
                <h2>{method}</h2>
                <p>Recommended financing method based on cost ratio 
                   and demand risk profile</p>
            </div>
            <div class="justification">
                {rationale}
            </div>
        </div>

        <div class="section">
            <h2>Financing Method Comparison</h2>
            <table>
                <tr>
                    <th>Method</th>
                    <th>Best When</th>
                    <th>Retailer Risk</th>
                    <th>Selected</th>
                </tr>
                <tr>
                    <td>Early Payment</td>
                    <td>Cost ratio &lt; 0.40</td>
                    <td>High</td>
                    <td>{"✅" if method == "Early Payment" else ""}</td>
                </tr>
                <tr>
                    <td>In-House Factoring</td>
                    <td>Cost ratio 0.40 – 0.70</td>
                    <td>Medium</td>
                    <td>{"✅" if method == "In-House Factoring" else ""}</td>
                </tr>
                <tr>
                    <td>Bank Financing</td>
                    <td>Cost ratio &gt; 0.70</td>
                    <td>Low</td>
                    <td>{"✅" if method == "Bank Financing" else ""}</td>
                </tr>
            </table>
        </div>

            <div class="footer">
                <p>Reference: Chen, X., Lu, Q., & Cai, G. (2020). Buyer Financing in Pull 
                Supply Chains. <em>Production and Operations Management.</em></p>
                <p>Generated by AI-Powered Supply Chain Finance Advisor</p>
            </div>
        </body>
        </html>
        """
        return html
    html_report = generate_html_report(
        retail_price, production_cost, cost_ratio,
        mean_demand, cv, risk_level, var_95,
        method, rationale, bias
    )

    st.download_button(
        label=" Download Full Report (HTML)",
        data=html_report,
        file_name="financing_recommendation.html",
        mime="text/html"
    )
    


    st.divider()
    st.caption(
        "Reference: Chen, X., Lu, Q., & Cai, G. (2020). Buyer "
        "Financing in Pull Supply Chains: Zero-Interest Early "
        "Payment or In-House Factoring? "
        "*Production and Operations Management.*"
    )

else:
    st.info(
        "Please upload a CSV file with your historical sales "
        "data to get started. The file should have two columns: "
        "'date' and 'demand'."
    )
    st.markdown("### Expected CSV format:")
    st.code("""date,demand
2021-01-01,105
2021-01-02,99
2021-01-03,107""")
    st.markdown(
        "You can use the sample data from the project's "
        "`outputs/forecast_results.csv` to test the dashboard."
    )