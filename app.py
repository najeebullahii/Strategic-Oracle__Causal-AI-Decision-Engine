import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Strategic Oracle | Causal AI Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"], p, div, span, label {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #f1f5f9;
    }

    .stApp {
        background: #060d1f;
    }

    #MainMenu, footer, header { visibility: hidden; }

    [data-testid="stSidebar"] {
        background: #0b1629 !important;
        border-right: 1px solid #1e3a5f;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        padding: 6px 0 !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        color: #38bdf8 !important;
    }

    .card {
        background: #0f1e35;
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 26px 22px;
        text-align: center;
    }

    .card-value {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 8px;
    }

    .card-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: #94a3b8;
    }

    .card-sub {
        font-size: 0.72rem;
        color: #475569;
        margin-top: 5px;
    }

    .page-title {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2.8rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .page-subtitle {
        font-size: 1rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 32px;
    }

    .sec-title {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 4px;
    }

    .sec-sub {
        font-size: 0.83rem;
        color: #64748b;
        margin-bottom: 20px;
    }

    .info-box {
        background: #0f1e35;
        border-left: 3px solid #3b82f6;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin: 14px 0;
        font-size: 0.92rem;
        color: #cbd5e1;
        line-height: 1.7;
    }

    .divline {
        height: 1px;
        background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
        margin: 28px 0;
    }

    .result-box {
        background: #0f1e35;
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 12px;
    }

    .result-num {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1;
    }

    .result-lbl {
        font-size: 0.75rem;
        color: #64748b;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 6px;
    }

    .stSlider label, .stNumberInput label, .stSelectbox label {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    .stMetric label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    .stMetric [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    .stTable th, .stTable td {
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
    }

    .streamlit-expanderHeader,
    [data-testid="stExpander"] details summary {
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    [data-testid="stExpander"] details summary div > span:first-child {
        display: none !important;
    }


    .stCaption {
        color: #475569 !important;
        font-size: 0.78rem !important;
    }
</style>
""", unsafe_allow_html=True)


# causal findings from phase 2 and 3
ATE            = 0.0681
BASELINE_RATE  = 0.0578
RAW_DIFF       = 0.0914
BIAS_REMOVED   = RAW_DIFF - ATE

refutation_results = {
    "Placebo Treatment":    {"new_effect": 0.0007, "p_value": 0.411, "expected": "~0.000"},
    "Random Common Cause":  {"new_effect": 0.0681, "p_value": 0.159, "expected": "~0.068"},
    "Data Subset":          {"new_effect": 0.0677, "p_value": 0.419, "expected": "~0.068"},
    "Bootstrap":            {"new_effect": 0.0682, "p_value": 0.481, "expected": "~0.068"},
}

@st.cache_data
def load_data():
    return pd.read_csv("bank-full-cleaned.csv")

df = load_data()
total_customers = len(df)


# sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding: 10px 0 28px 0;'>
        <div style='font-family: Outfit, sans-serif; font-size: 1.4rem; font-weight: 800; color: #38bdf8;'>
            Strategic Oracle
        </div>
        <div style='font-size: 0.72rem; color: #64748b; margin-top: 4px;
                    letter-spacing: 0.1em; text-transform: uppercase;'>
            Causal AI Decision Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px; background:#1e3a5f; margin-bottom:20px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px; font-weight:600;'>Navigation</div>", unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Executive Summary", "Bias Discovery", "What-If Simulator", "Validation Tests"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:1px; background:#1e3a5f; margin: 20px 0;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.78rem; color:#64748b; line-height:2;'>
        <span style='color:#94a3b8; font-weight:600;'>Dataset</span><br>
        UCI Bank Marketing<br>
        45,211 records<br><br>
        <span style='color:#94a3b8; font-weight:600;'>Method</span><br>
        DoWhy Causal Inference<br>
        Propensity Score Stratification<br><br>
        <span style='color:#94a3b8; font-weight:600;'>Validation</span><br>
        4 Refutation Tests<br>
        All p-values &gt; 0.05
    </div>
    """, unsafe_allow_html=True)


# ---- EXECUTIVE SUMMARY ----
if page == "Executive Summary":

    st.markdown("<div class='page-title'>Strategic Oracle</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Causal AI Decision Engine — Bank Marketing Campaign Analysis</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class='card'>
            <div class='card-value' style='color:#38bdf8;'>{total_customers:,}</div>
            <div class='card-label'>Total Customers</div>
            <div class='card-sub'>45K+ records analyzed</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='card'>
            <div class='card-value' style='color:#fbbf24;'>{RAW_DIFF:.1%}</div>
            <div class='card-label'>Raw Difference</div>
            <div class='card-sub'>Biased — before correction</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='card'>
            <div class='card-value' style='color:#34d399;'>{ATE:.1%}</div>
            <div class='card-label'>True Causal Effect</div>
            <div class='card-sub'>After bias removal (ATE)</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='card'>
            <div class='card-value' style='color:#f87171;'>{BIAS_REMOVED:.1%}</div>
            <div class='card-label'>Bias Detected</div>
            <div class='card-sub'>Selection bias removed</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>What This Project Proves</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>The difference between a data reporter and a business strategist</div>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class='info-box' style='border-left-color:#fbbf24;'>
            <b style='color:#fbbf24;'>The standard analysis view</b><br><br>
            A standard analyst looks at the 14.92% vs 5.78% subscription gap and concludes
            cellular calling drives more sales. That number goes into the campaign report —
            and it's the wrong number to act on.
            <br><br>
            <span style='color:#64748b; font-size:0.82rem;'>Acting on this inflates projected ROI
            and leads to budget being allocated in the wrong places.</span>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class='info-box' style='border-left-color:#34d399;'>
            <b style='color:#34d399;'>What the causal model found</b><br><br>
            Wealthier, younger customers naturally own cellular phones and naturally invest more.
            The bank was already reaching its best customers. Once you strip that out, the call
            itself only moves the needle by <b>6.81%</b> — still significant, but a very different
            budget conversation.
            <br><br>
            <span style='color:#64748b; font-size:0.82rem;'>Validated by four independent
            refutation tests. This is the number worth building strategy on.</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>Dataset Snapshot</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>UCI Bank Marketing — 45,211 records, 19 features after cleaning</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        tc = df['treatment'].value_counts().reset_index()
        tc.columns = ['Contact', 'Count']
        tc['Contact'] = tc['Contact'].map({1: 'Cellular', 0: 'Other'})
        fig = go.Figure(go.Pie(
            labels=tc['Contact'], values=tc['Count'], hole=0.62,
            marker=dict(colors=['#38bdf8', '#1e3a5f'], line=dict(color='#060d1f', width=2)),
            textinfo='label+percent', textfont=dict(size=13, color='#f1f5f9')
        ))
        fig.update_layout(
            title=dict(text="Contact Method Split", font=dict(color='#94a3b8', size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, height=270, margin=dict(t=40, b=10, l=10, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        oc = df['outcome'].value_counts().reset_index()
        oc.columns = ['Subscribed', 'Count']
        oc['Subscribed'] = oc['Subscribed'].map({1: 'Subscribed', 0: 'Did Not'})
        fig2 = go.Figure(go.Pie(
            labels=oc['Subscribed'], values=oc['Count'], hole=0.62,
            marker=dict(colors=['#34d399', '#1e3a5f'], line=dict(color='#060d1f', width=2)),
            textinfo='label+percent', textfont=dict(size=13, color='#f1f5f9')
        ))
        fig2.update_layout(
            title=dict(text="Overall Subscription Rate", font=dict(color='#94a3b8', size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, height=270, margin=dict(t=40, b=10, l=10, r=10)
        )
        st.plotly_chart(fig2, use_container_width=True)


# ---- BIAS DISCOVERY ----
elif page == "Bias Discovery":

    st.markdown("<div class='page-title'>Bias Discovery</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>How selection bias inflated the campaign's apparent effectiveness</div>", unsafe_allow_html=True)

    categories = ['Raw Difference<br>(Biased)', 'True Causal Effect<br>(Corrected)', 'Selection Bias<br>(Removed)']
    values     = [RAW_DIFF, ATE, BIAS_REMOVED]
    colors     = ['#fbbf24', '#34d399', '#f87171']

    fig = go.Figure()
    for cat, val, col in zip(categories, values, colors):
        fig.add_trace(go.Bar(
            x=[cat], y=[val * 100], name=cat,
            marker=dict(color=col, opacity=0.9, line=dict(color=col, width=1)),
            text=[f"{val:.2%}"], textposition='outside',
            textfont=dict(size=17, color=col, family='Outfit')
        ))

    fig.update_layout(
        title=dict(text="The Bias Breakdown — Raw vs True Causal Effect",
                   font=dict(color='#94a3b8', size=14)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, height=400,
        yaxis=dict(title="Effect Size (%)", color='#94a3b8',
                   gridcolor='rgba(255,255,255,0.05)', ticksuffix="%",
                   tickfont=dict(color='#94a3b8')),
        xaxis=dict(color='#94a3b8', tickfont=dict(color='#cbd5e1', size=12)),
        margin=dict(t=50, b=20, l=20, r=20), bargap=0.5
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Why the Bias Exists</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        The core issue is that cellular phone ownership isn't random. Younger, employed,
        financially stable customers are more likely to have a cellular number on file —
        and those same customers are also more likely to invest in a term deposit.
        <br><br>
        So when the raw data shows cellular contacts subscribing more, it's partly the call
        working and partly just the bank reaching a better segment of customers to begin with.
        DoWhy separates those two things by controlling for
        <b style='color:#38bdf8;'>age, job, education, marital status, account balance,
        housing loan, personal loan, and credit default history.</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Subscription Rate by Contact Method</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>The raw numbers — before causal correction</div>", unsafe_allow_html=True)

    sub_by = df.groupby('treatment')['outcome'].mean().reset_index()
    sub_by['Contact'] = sub_by['treatment'].map({1: 'Cellular', 0: 'Other'})
    sub_by['Rate']    = sub_by['outcome'] * 100

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=sub_by['Contact'], y=sub_by['Rate'],
        marker_color=['#38bdf8', '#1e3a5f'],
        text=[f"{r:.2f}%" for r in sub_by['Rate']],
        textposition='outside',
        textfont=dict(size=15, color='#f1f5f9', family='Outfit'),
        width=0.4
    ))
    fig3.add_hline(
        y=(ATE + BASELINE_RATE) * 100, line_dash="dash", line_color="#34d399",
        annotation_text=f"True causal ceiling: {(ATE + BASELINE_RATE)*100:.1f}%",
        annotation_font=dict(color='#34d399', size=11)
    )
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=340,
        yaxis=dict(title="Subscription Rate (%)", color='#94a3b8',
                   gridcolor='rgba(255,255,255,0.05)', ticksuffix="%",
                   tickfont=dict(color='#94a3b8')),
        xaxis=dict(color='#94a3b8', tickfont=dict(color='#cbd5e1', size=13)),
        margin=dict(t=20, b=20, l=20, r=20), showlegend=False
    )
    st.plotly_chart(fig3, use_container_width=True)


# ---- WHAT-IF SIMULATOR ----
elif page == "What-If Simulator":

    st.markdown("<div class='page-title'>What-If Simulator</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Run the numbers on a campaign before committing any budget</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        Adjust the parameters below to model different campaign scenarios. The engine runs two
        calculations side by side — what happens if the bank keeps doing what it's doing,
        versus what happens if it shifts fully to cellular outreach. The recommendation at the
        bottom updates automatically based on whether the switch actually makes financial sense.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Campaign Parameters</div>", unsafe_allow_html=True)

    ci1, ci2, ci3, ci4 = st.columns(4)

    with ci1:
        target_customers = st.number_input(
            "Customers to Call", min_value=1000, max_value=100000, value=10000, step=1000
        )
    with ci2:
        revenue_per_sub = st.slider("Revenue per Subscription ($)", 50, 2000, 250, 10)
    with ci3:
        cost_per_cellular = st.slider("Cost per Cellular Call ($)", 1.0, 10.0, 3.0, 0.5)
    with ci4:
        cost_per_standard = st.slider("Cost per Standard Call ($)", 0.5, 5.0, 1.0, 0.5)

    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)

    # run both scenarios
    baseline_subs     = int(target_customers * BASELINE_RATE)
    baseline_revenue  = baseline_subs * revenue_per_sub
    baseline_cost     = target_customers * cost_per_standard
    baseline_profit   = baseline_revenue - baseline_cost

    new_sub_rate      = BASELINE_RATE + ATE
    strategic_subs    = int(target_customers * new_sub_rate)
    strategic_revenue = strategic_subs * revenue_per_sub
    strategic_cost    = target_customers * cost_per_cellular
    strategic_profit  = strategic_revenue - strategic_cost

    added_subs        = strategic_subs - baseline_subs
    added_profit      = strategic_profit - baseline_profit
    extra_cost        = strategic_cost - baseline_cost

    st.markdown("<div class='sec-title'>Expected Campaign Outcomes</div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Expected Subscriptions", f"{strategic_subs:,}", f"+{added_subs:,} via Causal Strategy")
    with m2:
        st.metric("Total Strategy Profit", f"${strategic_profit:,.2f}", f"${added_profit:,.2f} vs Baseline")
    with m3:
        st.metric("Additional Investment Required", f"${extra_cost:,.2f}", delta_color="inverse")

    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Financial Comparison</div>", unsafe_allow_html=True)

    comparison_data = {
        "Metric": ["Total Cost", "Gross Revenue", "Net Profit"],
        "Status Quo (Standard Calls)":      [f"${baseline_cost:,.2f}",  f"${baseline_revenue:,.2f}",  f"${baseline_profit:,.2f}"],
        "Causal Strategy (Cellular Calls)": [f"${strategic_cost:,.2f}", f"${strategic_revenue:,.2f}", f"${strategic_profit:,.2f}"],
    }
    st.table(pd.DataFrame(comparison_data).set_index("Metric"))

    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Strategic Recommendation</div>", unsafe_allow_html=True)

    if added_profit > 0:
        st.success(
            f"Proceed. Investing an additional ${extra_cost:,.2f} in cellular outreach is projected "
            f"to return ${added_profit:,.2f} in net profit above the baseline. The numbers support the shift."
        )
    else:
        st.error(
            f"Hold. At the current call cost, switching to cellular results in a net loss of "
            f"${abs(added_profit):,.2f} compared to baseline. The economics don't work yet — "
            f"try reducing call cost or increasing the revenue-per-subscription assumption."
        )

    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Profit Projection Across Campaign Sizes</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>How net profit scales as more customers are contacted</div>", unsafe_allow_html=True)

    x_vals            = list(range(1000, 101000, 1000))
    baseline_profits  = [(int(x * BASELINE_RATE) * revenue_per_sub) - (x * cost_per_standard) for x in x_vals]
    strategic_profits = [(int(x * new_sub_rate) * revenue_per_sub) - (x * cost_per_cellular) for x in x_vals]

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(
        x=x_vals, y=baseline_profits, name="Status Quo",
        line=dict(color='#fbbf24', width=2, dash='dash')
    ))
    fig_proj.add_trace(go.Scatter(
        x=x_vals, y=strategic_profits, name="Causal Strategy",
        line=dict(color='#34d399', width=3),
        fill='tonexty', fillcolor='rgba(52,211,153,0.06)'
    ))
    fig_proj.add_trace(go.Scatter(
        x=[target_customers], y=[strategic_profit],
        mode='markers', name='Current Setting',
        marker=dict(color='#a78bfa', size=13, symbol='diamond',
                    line=dict(color='white', width=2))
    ))
    fig_proj.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=370,
        legend=dict(font=dict(color='#cbd5e1', size=12), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(title="Customers Called", color='#94a3b8',
                   gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#94a3b8')),
        yaxis=dict(title="Net Profit ($)", color='#94a3b8',
                   gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#94a3b8')),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_proj, use_container_width=True)
    st.caption("Powered by DoWhy Causal Inference · Refutation validated across placebo, random cause, subset, and bootstrap tests")


# ---- VALIDATION TESTS ----
elif page == "Validation Tests":

    st.markdown("<div class='page-title'>Validation Tests</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Four independent tests that confirm the causal finding holds up under pressure</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        Getting an ATE from a causal model is one thing. Trusting it is another.
        These four tests each attack the result from a different angle — replacing the treatment
        with noise, injecting fake variables, cutting the dataset, and resampling it. If the
        6.81% finding survives all four, it's not a fluke. A p-value above 0.05 on each test
        confirms the result is statistically stable.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Refutation Scorecard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>All four tests passed</div>", unsafe_allow_html=True)

    test_descriptions = {
        "Placebo Treatment":   "The real treatment column was replaced with random 1s and 0s. A valid causal model should find almost no effect — because random assignment can't cause subscriptions.",
        "Random Common Cause": "A completely fake variable was added as a confounder. If the model was fragile, this would shift the ATE. It didn't move.",
        "Data Subset":         "10% of the data was removed at random and the full analysis was rerun. The ATE barely changed, which means the finding isn't dependent on any specific slice of the data.",
        "Bootstrap":           "The dataset was resampled 20 times with replacement and the ATE recalculated each time. The average stayed consistent, confirming the result holds across different population samples.",
    }

    for test_name, results in refutation_results.items():
        p_val    = results["p_value"]
        new_eff  = results["new_effect"]
        expected = results["expected"]
        p_color  = "#34d399" if p_val > 0.05 else "#f87171"
        p_status = "Robust" if p_val > 0.05 else "Review"
        shift    = abs(new_eff - ATE)

        st.markdown(f"""
        <div style="background:#0f1e35; border:1px solid #1e3a5f; border-radius:14px;
                    padding:20px 24px; margin-bottom:14px;">

            <div style="display:flex; justify-content:space-between; align-items:center;
                        margin-bottom:18px;">
                <div style="font-family:Outfit,sans-serif; font-size:1.05rem; font-weight:700;
                            color:#f1f5f9;">{test_name}</div>
                <div style="background:{'rgba(52,211,153,0.12)' if p_val > 0.05 else 'rgba(248,113,113,0.12)'};
                            color:{p_color}; border:1px solid {p_color};
                            border-radius:20px; padding:3px 14px;
                            font-size:0.78rem; font-weight:600; letter-spacing:0.06em;">
                    {p_status}
                </div>
            </div>

            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:16px;">
                <div style="background:#060d1f; border-radius:10px; padding:16px; text-align:center;">
                    <div style="font-family:Outfit,sans-serif; font-size:2rem; font-weight:800;
                                color:{p_color};">{p_val:.3f}</div>
                    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em;
                                color:#64748b; margin-top:4px;">P-Value</div>
                    <div style="font-size:0.7rem; color:#475569; margin-top:3px;">
                        {'Above 0.05 — robust' if p_val > 0.05 else 'Below 0.05 — review'}
                    </div>
                </div>
                <div style="background:#060d1f; border-radius:10px; padding:16px; text-align:center;">
                    <div style="font-family:Outfit,sans-serif; font-size:2rem; font-weight:800;
                                color:#38bdf8;">{new_eff:.4f}</div>
                    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em;
                                color:#64748b; margin-top:4px;">New Effect</div>
                    <div style="font-size:0.7rem; color:#475569; margin-top:3px;">Expected {expected}</div>
                </div>
                <div style="background:#060d1f; border-radius:10px; padding:16px; text-align:center;">
                    <div style="font-family:Outfit,sans-serif; font-size:2rem; font-weight:800;
                                color:#a78bfa;">{shift:.4f}</div>
                    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em;
                                color:#64748b; margin-top:4px;">ATE Shift</div>
                    <div style="font-size:0.7rem; color:#475569; margin-top:3px;">{(shift/ATE)*100:.1f}% from original</div>
                </div>
            </div>

            <div style="background:#060d1f; border-left:3px solid #3b82f6; border-radius:0 8px 8px 0;
                        padding:12px 16px; font-size:0.88rem; color:#94a3b8; line-height:1.6;">
                {test_descriptions[test_name]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>P-Value Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>All values well above the 0.05 significance threshold</div>", unsafe_allow_html=True)

    test_names = list(refutation_results.keys())
    p_values   = [v['p_value'] for v in refutation_results.values()]

    fig_p = go.Figure()
    fig_p.add_hline(
        y=0.05, line_dash="dash", line_color="#f87171", line_width=2,
        annotation_text="0.05 threshold",
        annotation_font=dict(color='#f87171', size=11)
    )
    fig_p.add_trace(go.Bar(
        x=test_names, y=p_values,
        marker=dict(color=['#34d399' if p > 0.05 else '#f87171' for p in p_values], opacity=0.85),
        text=[f"{p:.3f}" for p in p_values], textposition='outside',
        textfont=dict(size=14, color='#f1f5f9', family='Outfit'),
        width=0.45
    ))
    fig_p.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        yaxis=dict(title="P-Value", color='#94a3b8',
                   gridcolor='rgba(255,255,255,0.05)', range=[0, 0.6],
                   tickfont=dict(color='#94a3b8')),
        xaxis=dict(color='#94a3b8', tickfont=dict(color='#cbd5e1', size=12)),
        margin=dict(t=30, b=20, l=20, r=20), showlegend=False
    )
    st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box' style='border-left-color:#34d399;'>
        The 6.81% Average Treatment Effect held up across all four tests. It wasn't sensitive
        to which rows were in the dataset, wasn't thrown off by a fake confounder, and collapsed
        to near zero when the real treatment was replaced with noise — exactly what a genuine
        causal effect should do. The finding is solid.
    </div>""", unsafe_allow_html=True)

    st.caption("DoWhy Causal Inference · UCI Bank Marketing Dataset · 45,211 records")



