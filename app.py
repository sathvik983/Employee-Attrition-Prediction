import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Set page layout and aesthetics
st.set_page_config(
    page_title="TalentGuard | Employee Attrition Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using glassmorphism and modern gradient designs
st.markdown("""
    <style>
    /* Clean layout changes */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Main title styling with gradient */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #FF4B4B, #852C8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.05em;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #7E8C9F;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Premium glassmorphic metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 75, 75, 0.4);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #8C9BAE;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Risk cards styling */
    .risk-high {
        background: rgba(255, 75, 75, 0.1);
        border: 1px solid rgba(255, 75, 75, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        color: #FF8A8A;
    }
    .risk-med {
        background: rgba(255, 165, 0, 0.1);
        border: 1px solid rgba(255, 165, 0, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        color: #FFD48A;
    }
    .risk-low {
        background: rgba(46, 204, 113, 0.1);
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        color: #8AFFB0;
    }
    
    /* Dynamic tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 8px 8px 0px 0px;
        color: #8C9BAE;
        font-weight: 600;
        padding: 0px 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(255, 75, 75, 0.15) 0%, rgba(255, 255, 255, 0) 100%);
        color: #FF4B4B !important;
        border-top: 2px solid #FF4B4B !important;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load model
@st.cache_resource
def load_model_pipeline():
    model_path = os.path.join("models", "best_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Helper function to load training results
@st.cache_data
def load_training_results():
    results_path = os.path.join("models", "training_results.json")
    if os.path.exists(results_path):
        import json
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

# Helper function to load clean data
@st.cache_data
def load_clean_data():
    clean_path = os.path.join("data", "clean_attrition.csv")
    if os.path.exists(clean_path):
        return pd.read_csv(clean_path)
    return None

# Load resources
pipeline = load_model_pipeline()
training_results = load_training_results()
df_clean = load_clean_data()

# Header layout
st.markdown('<div class="main-title">TALENTGUARD</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Driven Organizational Intelligence & Attrition Prediction System</div>', unsafe_allow_html=True)

if df_clean is None or pipeline is None:
    st.error("⚠️ Model files or datasets could not be loaded. Please run the training script (`python model_trainer.py`) to generate the artifacts.")
    st.stop()

# Prepare some metrics from the clean dataset
total_headcount = len(df_clean)
attrition_rate = (df_clean['Attrition'] == 1).mean() * 100
avg_monthly_income = df_clean['MonthlyIncome'].mean()
avg_tenure = df_clean['YearsAtCompany'].mean()

# Tabs
tab_eda, tab_sim, tab_importance, tab_batch = st.tabs([
    "📊 Organizational Analytics",
    "🔮 Individual Risk Predictor",
    "🔑 Strategic Risk Drivers",
    "🔍 Talent Scanning Directory"
])

# ----------------- TAB 1: ORGANIZATIONAL ANALYTICS -----------------
with tab_eda:
    # Key Stats Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_headcount:,}</div>
                <div class="metric-label">Total Headcount</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{attrition_rate:.1f}%</div>
                <div class="metric-label">Attrition Rate</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${avg_monthly_income:,.0f}</div>
                <div class="metric-label">Avg Monthly Income</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_tenure:.1f} Yrs</div>
                <div class="metric-label">Avg Company Tenure</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations Row 1
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Overtime vs. Attrition Rate")
        # Compute attrition rate by overtime
        ot_attrition = df_clean.groupby('OverTime')['Attrition'].mean().reset_index()
        ot_attrition['Attrition'] *= 100
        ot_attrition['OverTime'] = ot_attrition['OverTime'].map({'Yes': 'Working Overtime', 'No': 'No Overtime'})
        fig_ot = px.bar(
            ot_attrition, 
            x='OverTime', 
            y='Attrition',
            color='OverTime',
            color_discrete_map={'Working Overtime': '#FF4B4B', 'No Overtime': '#2ECC71'},
            labels={'Attrition': 'Attrition Rate (%)', 'OverTime': 'Overtime Status'},
            text_auto='.1f'
        )
        fig_ot.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=350,
            font=dict(color='white')
        )
        st.plotly_chart(fig_ot, use_container_width=True)
        st.caption("Employees working overtime are roughly 3x more likely to leave the company.")

    with col_chart2:
        st.subheader("Job Role Attrition Risk Breakdown")
        role_attrition = df_clean.groupby('JobRole')['Attrition'].mean().reset_index()
        role_attrition['Attrition'] *= 100
        role_attrition = role_attrition.sort_values(by='Attrition', ascending=True)
        fig_role = px.bar(
            role_attrition, 
            y='JobRole', 
            x='Attrition',
            orientation='h',
            color='Attrition',
            color_continuous_scale='Reds',
            labels={'Attrition': 'Attrition Rate (%)', 'JobRole': 'Job Role'}
        )
        fig_role.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            height=350,
            font=dict(color='white')
        )
        st.plotly_chart(fig_role, use_container_width=True)
        st.caption("Sales Representatives and Laboratory Technicians experience the highest attrition rates.")

    # Visualizations Row 2
    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        st.subheader("Monthly Income vs. Age Distribution")
        # Scatter plot colored by attrition
        df_plot = df_clean.copy()
        df_plot['Status'] = df_plot['Attrition'].map({1: 'Left Company (Attrition)', 0: 'Active Employee'})
        fig_scatter = px.scatter(
            df_plot,
            x='Age',
            y='MonthlyIncome',
            color='Status',
            color_discrete_map={'Left Company (Attrition)': '#FF4B4B', 'Active Employee': 'rgba(46, 204, 113, 0.4)'},
            labels={'MonthlyIncome': 'Monthly Income ($)', 'Age': 'Age (Years)'},
            opacity=0.8
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=380,
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Attrition clusters heavily among younger employees (under 35) with lower monthly incomes.")

    with col_chart4:
        st.subheader("Job Satisfaction vs. Attrition Rate")
        sat_attrition = df_clean.groupby('JobSatisfaction')['Attrition'].mean().reset_index()
        sat_attrition['Attrition'] *= 100
        sat_attrition['JobSatisfaction'] = sat_attrition['JobSatisfaction'].map({
            1: '1 - Low', 2: '2 - Medium', 3: '3 - High', 4: '4 - Very High'
        })
        fig_sat = px.bar(
            sat_attrition,
            x='JobSatisfaction',
            y='Attrition',
            color='Attrition',
            color_continuous_scale='OrRd',
            labels={'Attrition': 'Attrition Rate (%)', 'JobSatisfaction': 'Job Satisfaction Level'},
            text_auto='.1f'
        )
        fig_sat.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            height=380,
            font=dict(color='white')
        )
        st.plotly_chart(fig_sat, use_container_width=True)
        st.caption("Higher job satisfaction demonstrates a clear, incremental reduction in attrition rates.")

# ----------------- TAB 2: INDIVIDUAL RISK PREDICTOR -----------------
with tab_sim:
    st.subheader("Predict Attrition Probability for a Single Employee")
    st.write("Input the employee's parameters below to calculate their risk score.")
    
    # 3-Column form layout
    with st.form("individual_prediction_form"):
        col_dem, col_role, col_sat = st.columns(3)
        
        with col_dem:
            st.markdown("### 📊 Demographics & Compensation")
            age = st.slider("Age", 18, 65, 35)
            gender = st.selectbox("Gender", ["Female", "Male"])
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000, step=100)
            stock_level = st.slider("Stock Option Level", 0, 3, 1)
            percent_salary_hike = st.slider("Percent Salary Hike (%)", 11, 25, 14)
            
        with col_role:
            st.markdown("### 🏢 Job Role & Work History")
            dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            
            # Filter job roles depending on department to make it logical
            if dept == "Sales":
                roles = ["Sales Executive", "Sales Representative", "Manager"]
            elif dept == "Research & Development":
                roles = ["Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Research Director", "Manager"]
            else:
                roles = ["Human Resources", "Manager"]
                
            job_role = st.selectbox("Job Role", roles)
            job_level = st.slider("Job Level", 1, 5, 2)
            overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
            travel = st.selectbox("Business Travel Frequency", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            distance_from_home = st.slider("Distance From Home (miles)", 1, 30, 8)
            num_companies = st.slider("Num Companies Worked", 0, 9, 2)
            
        with col_sat:
            st.markdown("### 🤝 Engagement & Tenure")
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3, help="1=Low, 4=Very High")
            env_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
            rel_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
            work_life = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
            
            total_working_years = st.slider("Total Working Years", 0, 40, 10)
            years_at_company = st.slider("Years At Company", 0, total_working_years, 5)
            years_in_role = st.slider("Years In Current Role", 0, years_at_company, 3)
            years_since_promo = st.slider("Years Since Last Promotion", 0, years_at_company, 1)
            years_curr_manager = st.slider("Years With Current Manager", 0, years_at_company, 3)
            
            # Constant values to inject back
            daily_rate = 800
            hourly_rate = 65
            monthly_rate = 14000
            training_last_year = 2
            perf_rating = 3 if percent_salary_hike < 20 else 4
            
        # Submit
        submit_prediction = st.form_submit_button("⚡ Compute Attrition Risk", use_container_width=True)
        
    if submit_prediction:
        # Create input dataframe matching raw features
        input_data = {
            'Age': [age],
            'BusinessTravel': [travel],
            'DailyRate': [daily_rate],
            'Department': [dept],
            'DistanceFromHome': [distance_from_home],
            'Education': [3], # Defaulting average education
            'EducationField': ['Life Sciences'], # Defaulting
            'EnvironmentSatisfaction': [env_satisfaction],
            'Gender': [gender],
            'HourlyRate': [hourly_rate],
            'JobInvolvement': [3], # Defaulting
            'JobLevel': [job_level],
            'JobRole': [job_role],
            'JobSatisfaction': [job_satisfaction],
            'MaritalStatus': [marital],
            'MonthlyIncome': [monthly_income],
            'MonthlyRate': [monthly_rate],
            'NumCompaniesWorked': [num_companies],
            'OverTime': [overtime],
            'PercentSalaryHike': [percent_salary_hike],
            'PerformanceRating': [perf_rating],
            'RelationshipSatisfaction': [rel_satisfaction],
            'StockOptionLevel': [stock_level],
            'TotalWorkingYears': [total_working_years],
            'TrainingTimesLastYear': [training_last_year],
            'WorkLifeBalance': [work_life],
            'YearsAtCompany': [years_at_company],
            'YearsInCurrentRole': [years_in_role],
            'YearsSinceLastPromotion': [years_since_promo],
            'YearsWithCurrManager': [years_curr_manager]
        }
        
        df_input = pd.DataFrame(input_data)
        
        # Calculate engineered features using feature_engineering
        from feature_engineering import add_custom_features
        df_engineered = add_custom_features(df_input)
        
        # Make prediction
        proba = pipeline.predict_proba(df_engineered)[0][1]
        risk_pct = proba * 100
        
        # Display Result Card
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown("### Risk Level Indicator")
            # Gauge chart using plotly
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_pct,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Attrition Risk Probability (%)", 'font': {'size': 16, 'color': "white"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#FF4B4B" if risk_pct >= 60 else ("#FFA500" if risk_pct >= 30 else "#2ECC71")},
                    'bgcolor': "rgba(255, 255, 255, 0.05)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255, 255, 255, 0.2)",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.1)'},
                        {'range': [30, 60], 'color': 'rgba(255, 165, 0, 0.1)'},
                        {'range': [60, 100], 'color': 'rgba(255, 75, 75, 0.1)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=250,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with res_col2:
            st.markdown("### Diagnosis & Action Items")
            
            # Risk Level Callout
            if risk_pct >= 60:
                st.markdown(f"""
                    <div class="risk-high">
                        <h4>🚨 HIGH ATTRITION RISK ({risk_pct:.1f}%)</h4>
                        This employee shows significant indicators associated with high turnover. Immediate retention efforts are strongly advised.
                    </div>
                """, unsafe_allow_html=True)
            elif risk_pct >= 30:
                st.markdown(f"""
                    <div class="risk-med">
                        <h4>⚠️ MEDIUM ATTRITION RISK ({risk_pct:.1f}%)</h4>
                        This employee has several mild risk indicators. Regular check-ins and job alignment discussion are recommended.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="risk-low">
                        <h4>✅ LOW ATTRITION RISK ({risk_pct:.1f}%)</h4>
                        This employee shows strong retention signals and is highly likely to stay. Maintain current levels of engagement.
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Explainable AI risk factors
            drivers = []
            if overtime == 'Yes':
                drivers.append("🔴 **Overtime Overload**: The employee works overtime, which is the strongest driver of organizational attrition.")
            if job_satisfaction <= 2:
                drivers.append("🔴 **Low Job Satisfaction**: Indicated a score of 2 or less, reflecting potential role mismatch or burnout.")
            if env_satisfaction <= 2:
                drivers.append("🔴 **Low Work Environment Satisfaction**: Unhappy with workspace atmosphere or culture.")
            if years_since_promo >= 3:
                drivers.append(f"🔴 **Career Stagnation**: Has not received a promotion in {years_since_promo} years, indicating a lack of clear progression.")
            if monthly_income < 3500:
                drivers.append("🔴 **Below-Average Compensation**: Monthly income is lower than the typical industry median for retention.")
            if stock_level == 0:
                drivers.append("🔴 **No Equity Incentive**: Stock option level is set to 0, which reduces financial lock-in.")
            if distance_from_home > 15:
                drivers.append(f"🔴 **Long Commute**: Located {distance_from_home} miles away from the office, raising travel fatigue risks.")
            
            if not drivers:
                st.success("No critical risk drivers identified! The employee exhibits healthy tenure and satisfaction parameters.")
            else:
                st.write("**Identified Risk Drivers:**")
                for d in drivers:
                    st.markdown(d)
                    
            # Retention Strategy Action Items
            st.write("")
            st.write("**Suggested Retention Actions:**")
            if overtime == 'Yes':
                st.info("💡 **Action**: Review workload. Can tasks be delegated? Ensure they are compensated or recognized for overtime.")
            if job_satisfaction <= 2 or years_since_promo >= 3:
                st.info("💡 **Action**: Hold a career development conversation. Set clear benchmarks for the next promotion path.")
            if stock_level == 0 and monthly_income < 6000:
                st.info("💡 **Action**: Review salary package or consider adding a performance bonus / stock options in the next cycle.")

# ----------------- TAB 3: STRATEGIC RISK DRIVERS -----------------
with tab_importance:
    st.subheader("What Drives Attrition Across the Organization?")
    st.write("Below are the feature importances extracted from the trained Random Forest model. These show which factors carry the most predictive weight when determining if an employee will stay or leave.")
    
    importance_path = os.path.join("models", "feature_importance.csv")
    if os.path.exists(importance_path):
        df_imp = pd.read_csv(importance_path)
        
        # Display top 15 features
        df_imp_top = df_imp.head(15).copy()
        
        # Give cleaner feature labels
        feature_labels = {
            'MonthlyIncome': 'Monthly Income',
            'Age': 'Employee Age',
            'TotalWorkingYears': 'Total Work Experience (Years)',
            'SatisfactionSum': 'Overall Satisfaction Score (Custom)',
            'YearsAtCompany': 'Years at Current Company',
            'OverTime_Yes': 'Works Overtime',
            'OverTime_No': 'Does Not Work Overtime',
            'YearsWithCurrManager': 'Years under Current Manager',
            'IncomePerAge': 'Income-to-Age Ratio (Custom)',
            'StockOptionLevel': 'Stock Option Level',
            'DailyRate': 'Daily Billing Rate',
            'DistanceFromHome': 'Distance from Home (Miles)',
            'JobLevel': 'Job Level Rank',
            'MonthlyRate': 'Monthly Rate Metric',
            'HourlyRate': 'Hourly Rate Metric',
            'YearsAtCompanyPerTotalYears': 'Company Tenure Ratio (Custom)',
            'YearsInCurrentRolePerYearsAtCompany': 'Role Stability Ratio (Custom)',
            'NumCompaniesWorked': 'Number of Companies Worked',
            'YearsInCurrentRole': 'Years in Current Role'
        }
        
        df_imp_top['FeatureLabel'] = df_imp_top['Feature'].map(lambda x: feature_labels.get(x, x))
        
        col_imp1, col_imp2 = st.columns([3, 2])
        
        with col_imp1:
            fig_imp = px.bar(
                df_imp_top,
                y='FeatureLabel',
                x='Value',
                orientation='h',
                color='Value',
                color_continuous_scale='Viridis',
                labels={'Value': 'Relative Importance (Weight)', 'FeatureLabel': 'Feature Name'}
            )
            fig_imp.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                height=450,
                font=dict(color='white')
            )
            fig_imp.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with col_imp2:
            st.markdown("### 📋 Executive Recommendations")
            st.markdown("""
                Based on our predictive models, HR leadership should focus on three primary retention pillars:
                
                1. **Overtime Auditing & Caps**
                   * *Finding*: Working Overtime is a top-5 predictor and multiplies attrition risk by 3x.
                   * *Policy*: Institute a weekly cap on overtime hours, particularly in high-churn roles like *Sales Representatives* and *Laboratory Technicians*.
                
                2. **Compensation & Equity Lock-Ins**
                   * *Finding*: `MonthlyIncome` and `StockOptionLevel` are dominant predictive factors.
                   * *Policy*: Benchmark salaries for mid-level managers. Ensure all employees at Level 1-2 have baseline stock option incentive programs.
                
                3. **Career Tenures & Manager Fit**
                   * *Finding*: The custom features `SatisfactionSum` (overall satisfaction) and tenure under the current manager are strong predictors of exit.
                   * *Policy*: Run mandatory bi-annual feedback loops for supervisors whose teams exhibit higher attrition probabilities. Introduce rotating internal job transfers for employees exceeding 3 years in the same role without promotion.
            """)
    else:
        st.warning("Feature importance CSV not found. Please re-run training.")

# ----------------- TAB 4: TALENT SCANNING DIRECTORY -----------------
with tab_batch:
    st.subheader("Talent Registry Attrition Scanner")
    st.write("Scan and screen all employees in the current database. The list below is sorted by predicted attrition risk, showing the most vulnerable employees at the top.")
    
    # Run batch prediction on all clean data
    @st.cache_data
    def run_batch_prediction(_pipeline, df):
        # We need to compute custom features first
        from feature_engineering import add_custom_features
        df_eng = add_custom_features(df)
        
        # Run prediction
        probas = _pipeline.predict_proba(df_eng)[:, 1]
        
        df_result = df.copy()
        df_result['Risk Probability (%)'] = probas * 100
        
        # Categorize
        def get_risk_cat(p):
            if p >= 60: return "🔴 High"
            elif p >= 30: return "⚠️ Medium"
            return "✅ Low"
        df_result['Risk Category'] = df_result['Risk Probability (%)'].map(get_risk_cat)
        
        # Sort by risk descending
        df_result = df_result.sort_values(by='Risk Probability (%)', ascending=False)
        return df_result

    df_batch = run_batch_prediction(pipeline, df_clean)
    
    # Filter controls row
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dept_filter = st.multiselect("Filter by Department", df_batch['Department'].unique())
    with col_f2:
        role_filter = st.multiselect("Filter by Job Role", df_batch['JobRole'].unique())
    with col_f3:
        risk_filter = st.multiselect("Filter by Risk Category", ["🔴 High", "⚠️ Medium", "✅ Low"])
        
    # Apply filters
    df_filtered = df_batch.copy()
    if dept_filter:
        df_filtered = df_filtered[df_filtered['Department'].isin(dept_filter)]
    if role_filter:
        df_filtered = df_filtered[df_filtered['JobRole'].isin(role_filter)]
    if risk_filter:
        df_filtered = df_filtered[df_filtered['RiskCategory' if 'RiskCategory' in df_filtered.columns else 'Risk Category'].isin(risk_filter)]
        
    # Clean table for display
    display_cols = [
        'Age', 'Gender', 'Department', 'JobRole', 'MonthlyIncome', 
        'OverTime', 'YearsAtCompany', 'YearsSinceLastPromotion', 
        'Risk Category', 'Risk Probability (%)'
    ]
    
    # Format and present
    df_show = df_filtered[display_cols].copy()
    df_show['MonthlyIncome'] = df_show['MonthlyIncome'].map(lambda x: f"${x:,.0f}")
    df_show['Risk Probability (%)'] = df_show['Risk Probability (%)'].map(lambda x: f"{x:.1f}%")
    
    st.write(f"Showing {len(df_show)} employees:")
    st.dataframe(
        df_show, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Risk Category": st.column_config.TextColumn("Risk Category", width="medium"),
            "Risk Probability (%)": st.column_config.ProgressColumn("Risk Probability (%)", format="%f", min_value=0.0, max_value=100.0)
        }
    )
