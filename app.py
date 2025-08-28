import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import scipy.stats as stats
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Set page config for scientific dashboard
st.set_page_config(
    page_title="Scientific Results Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for scientific styling
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .scientific-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .experiment-section {
        border-left: 4px solid #3b82f6;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .statistical-summary {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Generate scientific experimental data
@st.cache_data
def generate_experimental_data():
    np.random.seed(42)
    
    # Experiment parameters
    n_samples = 200
    n_groups = 4
    n_timepoints = 10
    
    # Generate experimental groups data
    groups = ['Control', 'Treatment A', 'Treatment B', 'Treatment C']
    conditions = np.repeat(groups, n_samples // n_groups)
    
    # Generate response data with different effect sizes
    base_response = np.random.normal(100, 15, n_samples)
    treatment_effects = {'Control': 0, 'Treatment A': 12, 'Treatment B': 8, 'Treatment C': -5}
    
    responses = []
    group_labels = []
    
    for i, condition in enumerate(conditions):
        effect = treatment_effects[condition]
        noise = np.random.normal(0, 10)
        response = base_response[i] + effect + noise
        responses.append(response)
        group_labels.append(condition)
    
    experimental_df = pd.DataFrame({
        'Group': group_labels,
        'Response': responses,
        'Subject_ID': range(1, n_samples + 1),
        'Age': np.random.normal(35, 12, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Baseline': np.random.normal(95, 10, n_samples)
    })
    
    # Generate time series data
    time_points = range(0, n_timepoints)
    time_series_data = []
    
    for group in groups:
        for t in time_points:
            base_value = 50
            trend = treatment_effects[group] * (t / n_timepoints)
            noise = np.random.normal(0, 5)
            
            for rep in range(3):  # 3 replicates per timepoint
                value = base_value + trend + noise + np.random.normal(0, 2)
                time_series_data.append({
                    'Time': t,
                    'Group': group,
                    'Value': value,
                    'Replicate': rep + 1
                })
    
    time_series_df = pd.DataFrame(time_series_data)
    
    # Generate correlation data
    n_genes = 1000
    gene_expression = pd.DataFrame({
        'Gene_ID': [f'Gene_{i:04d}' for i in range(n_genes)],
        'Expression_Control': np.random.lognormal(2, 1, n_genes),
        'Expression_Treatment': np.random.lognormal(2.2, 1.1, n_genes),
        'P_value': np.random.beta(0.1, 2, n_genes),
        'Log2_FC': np.random.normal(0.5, 1.5, n_genes)
    })
    
    # Add significance classification
    gene_expression['Significant'] = gene_expression['P_value'] < 0.05
    gene_expression['Regulation'] = np.where(
        gene_expression['Log2_FC'] > 1, 'Upregulated',
        np.where(gene_expression['Log2_FC'] < -1, 'Downregulated', 'No Change')
    )
    
    return experimental_df, time_series_df, gene_expression

# Load experimental data
experimental_df, time_series_df, gene_expression = generate_experimental_data()

# Header
st.markdown("""
<div class="scientific-header">
    <h1>🔬 Scientific Results Dashboard</h1>
    <p>Comprehensive Analysis of Experimental Data and Research Findings</p>
</div>
""", unsafe_allow_html=True)

# Study Information
with st.expander("📋 Study Information", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Study Title:** Comparative Analysis of Novel Therapeutic Interventions")
        st.write("**Principal Investigator:** Dr. Jane Smith")
        st.write("**Institution:** Research University Medical Center")
    
    with col2:
        st.write("**Study Period:** January 2024 - June 2024")
        st.write("**Sample Size:** 200 subjects")
        st.write("**Study Design:** Randomized Controlled Trial")
    
    with col3:
        st.write("**Primary Endpoint:** Treatment Response")
        st.write("**Statistical Power:** 0.80 (α = 0.05)")
        st.write("**Status:** Analysis Complete")

# Key Statistical Metrics
st.subheader("📊 Key Statistical Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

# Calculate key statistics
control_mean = experimental_df[experimental_df['Group'] == 'Control']['Response'].mean()
treatment_a_mean = experimental_df[experimental_df['Group'] == 'Treatment A']['Response'].mean()
effect_size = (treatment_a_mean - control_mean) / experimental_df[experimental_df['Group'] == 'Control']['Response'].std()

# Perform ANOVA
f_stat, p_value = stats.f_oneway(
    experimental_df[experimental_df['Group'] == 'Control']['Response'],
    experimental_df[experimental_df['Group'] == 'Treatment A']['Response'],
    experimental_df[experimental_df['Group'] == 'Treatment B']['Response'],
    experimental_df[experimental_df['Group'] == 'Treatment C']['Response']
)

with col1:
    st.metric(
        label="Sample Size (n)",
        value=len(experimental_df)
    )

with col2:
    st.metric(
        label="Effect Size (Cohen's d)",
        value=f"{effect_size:.3f}",
        delta="Large effect" if abs(effect_size) > 0.8 else "Medium effect" if abs(effect_size) > 0.5 else "Small effect"
    )

with col3:
    st.metric(
        label="ANOVA F-statistic",
        value=f"{f_stat:.2f}",
        delta=f"p = {p_value:.4f}"
    )

with col4:
    st.metric(
        label="Significant Genes",
        value=f"{gene_expression['Significant'].sum()}",
        delta=f"{(gene_expression['Significant'].sum()/len(gene_expression)*100):.1f}%"
    )

with col5:
    st.metric(
        label="R² (Correlation)",
        value=f"{np.corrcoef(gene_expression['Expression_Control'], gene_expression['Expression_Treatment'])[0,1]**2:.3f}",
        delta="Strong correlation"
    )

st.markdown("---")

# Main Results Section
st.subheader("🎯 Primary Results")

# First row - Main experimental results
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Treatment Group Comparison")
    
    # Box plot for group comparison
    fig_box = px.box(
        experimental_df, 
        x='Group', 
        y='Response',
        title="Response by Treatment Group",
        color='Group',
        points="outliers"
    )
    fig_box.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Treatment Group",
        yaxis_title="Response (AU)"
    )
    
    # Add statistical annotations
    fig_box.add_annotation(
        x=1, y=experimental_df['Response'].max() * 1.1,
        text=f"ANOVA p-value: {p_value:.4f}",
        showarrow=False,
        font=dict(size=12, color="red" if p_value < 0.05 else "black")
    )
    
    st.plotly_chart(fig_box, use_container_width=True)

with col2:
    st.markdown("#### Statistical Distribution Analysis")
    
    # Violin plot with statistical details
    fig_violin = px.violin(
        experimental_df,
        x='Group',
        y='Response',
        title="Distribution Analysis with Density",
        box=True,
        color='Group'
    )
    fig_violin.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Treatment Group",
        yaxis_title="Response (AU)"
    )
    st.plotly_chart(fig_violin, use_container_width=True)

# Second row - Time series and correlation analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Temporal Response Profile")
    
    # Time series with confidence intervals
    time_summary = time_series_df.groupby(['Time', 'Group']).agg({
        'Value': ['mean', 'std', 'count']
    }).reset_index()
    time_summary.columns = ['Time', 'Group', 'Mean', 'Std', 'Count']
    time_summary['SEM'] = time_summary['Std'] / np.sqrt(time_summary['Count'])
    time_summary['CI_lower'] = time_summary['Mean'] - 1.96 * time_summary['SEM']
    time_summary['CI_upper'] = time_summary['Mean'] + 1.96 * time_summary['SEM']
    
    fig_time = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, group in enumerate(['Control', 'Treatment A', 'Treatment B', 'Treatment C']):
        group_data = time_summary[time_summary['Group'] == group]
        
        # Add confidence interval
        fig_time.add_trace(go.Scatter(
            x=list(group_data['Time']) + list(group_data['Time'][::-1]),
            y=list(group_data['CI_upper']) + list(group_data['CI_lower'][::-1]),
            fill='tonexty' if i == 0 else 'tonexty',
            fillcolor=f'rgba({int(mcolors.to_rgba(colors[i])[0]*255)},{int(mcolors.to_rgba(colors[i])[1]*255)},{int(mcolors.to_rgba(colors[i])[2]*255)},0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        # Add mean line
        fig_time.add_trace(go.Scatter(
            x=group_data['Time'],
            y=group_data['Mean'],
            mode='lines+markers',
            name=group,
            line=dict(color=colors[i], width=3),
            marker=dict(size=6)
        ))
    
    fig_time.update_layout(
        title="Temporal Response with 95% Confidence Intervals",
        xaxis_title="Time Point",
        yaxis_title="Response Value",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_time, use_container_width=True)

with col2:
    st.markdown("#### Gene Expression Analysis (Volcano Plot)")
    
    # Volcano plot
    gene_expression['neg_log10_p'] = -np.log10(gene_expression['P_value'])
    
    fig_volcano = px.scatter(
        gene_expression,
        x='Log2_FC',
        y='neg_log10_p',
        color='Regulation',
        title="Differential Gene Expression",
        color_discrete_map={
            'Upregulated': '#d62728',
            'Downregulated': '#2ca02c',
            'No Change': '#7f7f7f'
        },
        opacity=0.6
    )
    
    # Add significance threshold lines
    fig_volcano.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red", 
                         annotation_text="p = 0.05")
    fig_volcano.add_vline(x=1, line_dash="dash", line_color="red")
    fig_volcano.add_vline(x=-1, line_dash="dash", line_color="red")
    
    fig_volcano.update_layout(
        height=400,
        xaxis_title="Log2 Fold Change",
        yaxis_title="-log10(p-value)"
    )
    st.plotly_chart(fig_volcano, use_container_width=True)

st.markdown("---")

# Advanced Analysis Section
st.subheader("🔍 Advanced Statistical Analysis")

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Regression Analysis", 
    "🎲 Power Analysis", 
    "📊 Correlation Matrix",
    "🧮 Detailed Statistics"
])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Linear Regression: Age vs Response")
        
        fig_regression = px.scatter(
            experimental_df,
            x='Age',
            y='Response',
            color='Group',
            trendline='ols',
            title="Age-Response Relationship by Group"
        )
        fig_regression.update_layout(height=350)
        st.plotly_chart(fig_regression, use_container_width=True)
        
        # Calculate R-squared
        correlation = np.corrcoef(experimental_df['Age'], experimental_df['Response'])[0,1]
        r_squared = correlation**2
        st.info(f"**R² = {r_squared:.3f}** | **Correlation = {correlation:.3f}**")
    
    with col2:
        st.markdown("#### Baseline vs Response Correlation")
        
        fig_baseline = px.scatter(
            experimental_df,
            x='Baseline',
            y='Response',
            color='Group',
            size='Age',
            title="Baseline vs Final Response",
            trendline='ols'
        )
        fig_baseline.update_layout(height=350)
        st.plotly_chart(fig_baseline, use_container_width=True)
        
        baseline_corr = np.corrcoef(experimental_df['Baseline'], experimental_df['Response'])[0,1]
        st.info(f"**Baseline Correlation = {baseline_corr:.3f}**")

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Power Analysis Results")
        
        # Simulate power analysis
        effect_sizes = np.arange(0.1, 2.0, 0.1)
        sample_sizes = [20, 50, 100, 200]
        
        power_data = []
        for n in sample_sizes:
            for effect in effect_sizes:
                # Simplified power calculation
                power = 1 - stats.norm.cdf(1.96 - effect * np.sqrt(n/2))
                power_data.append({
                    'Effect_Size': effect,
                    'Sample_Size': n,
                    'Power': power
                })
        
        power_df = pd.DataFrame(power_data)
        
        fig_power = px.line(
            power_df,
            x='Effect_Size',
            y='Power',
            color='Sample_Size',
            title="Statistical Power vs Effect Size",
            labels={'Sample_Size': 'Sample Size (n)'}
        )
        fig_power.add_hline(y=0.8, line_dash="dash", line_color="red",
                           annotation_text="Power = 0.80")
        fig_power.update_layout(height=350)
        st.plotly_chart(fig_power, use_container_width=True)
    
    with col2:
        st.markdown("#### Effect Size Interpretation")
        
        # Effect size classification
        effect_sizes_calc = []
        for group in ['Treatment A', 'Treatment B', 'Treatment C']:
            control_data = experimental_df[experimental_df['Group'] == 'Control']['Response']
            treatment_data = experimental_df[experimental_df['Group'] == group]['Response']
            
            cohen_d = (treatment_data.mean() - control_data.mean()) / np.sqrt(
                ((len(treatment_data) - 1) * treatment_data.var() + 
                 (len(control_data) - 1) * control_data.var()) / 
                (len(treatment_data) + len(control_data) - 2)
            )
            
            effect_sizes_calc.append({
                'Comparison': f'{group} vs Control',
                'Cohen_d': cohen_d,
                'Magnitude': 'Large' if abs(cohen_d) > 0.8 else 'Medium' if abs(cohen_d) > 0.5 else 'Small'
            })
        
        effect_df = pd.DataFrame(effect_sizes_calc)
        
        fig_effect = px.bar(
            effect_df,
            x='Comparison',
            y='Cohen_d',
            color='Magnitude',
            title="Effect Sizes (Cohen's d)",
            color_discrete_map={'Large': '#d62728', 'Medium': '#ff7f0e', 'Small': '#2ca02c'}
        )
        fig_effect.add_hline(y=0.8, line_dash="dash", annotation_text="Large effect")
        fig_effect.add_hline(y=0.5, line_dash="dash", annotation_text="Medium effect")
        fig_effect.add_hline(y=0.2, line_dash="dash", annotation_text="Small effect")
        fig_effect.update_layout(height=350)
        st.plotly_chart(fig_effect, use_container_width=True)

with tab3:
    st.markdown("#### Correlation Matrix Analysis")
    
    # Create correlation matrix
    numeric_cols = ['Response', 'Age', 'Baseline']
    corr_matrix = experimental_df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Continuous Variables",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Pairplot equivalent
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter_matrix = px.scatter_matrix(
            experimental_df,
            dimensions=numeric_cols,
            color='Group',
            title="Pairwise Relationships"
        )
        fig_scatter_matrix.update_layout(height=400)
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)
    
    with col2:
        st.markdown("#### Correlation Significance Tests")
        
        correlation_tests = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Only upper triangle
                    corr_coef, p_val = stats.pearsonr(experimental_df[col1], experimental_df[col2])
                    correlation_tests.append({
                        'Variables': f'{col1} vs {col2}',
                        'Correlation': f'{corr_coef:.3f}',
                        'P-value': f'{p_val:.4f}',
                        'Significant': 'Yes' if p_val < 0.05 else 'No'
                    })
        
        corr_test_df = pd.DataFrame(correlation_tests)
        st.dataframe(corr_test_df, use_container_width=True)

with tab4:
    st.markdown("#### Comprehensive Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Descriptive Statistics by Group")
        
        descriptive_stats = experimental_df.groupby('Group')['Response'].agg([
            'count', 'mean', 'std', 'min', 'max', 
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 50),
            lambda x: np.percentile(x, 75)
        ]).round(3)
        
        descriptive_stats.columns = ['N', 'Mean', 'Std', 'Min', 'Max', 'Q1', 'Median', 'Q3']
        st.dataframe(descriptive_stats, use_container_width=True)
    
    with col2:
        st.markdown("##### Post-hoc Analysis (Tukey HSD)")
        
        # Simplified post-hoc results (normally would use statsmodels)
        groups = experimental_df['Group'].unique()
        posthoc_results = []
        
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if i < j:
                    data1 = experimental_df[experimental_df['Group'] == group1]['Response']
                    data2 = experimental_df[experimental_df['Group'] == group2]['Response']
                    
                    t_stat, p_val = stats.ttest_ind(data1, data2)
                    
                    posthoc_results.append({
                        'Comparison': f'{group1} vs {group2}',
                        't-statistic': f'{t_stat:.3f}',
                        'p-value': f'{p_val:.4f}',
                        'Significant': 'Yes' if p_val < 0.05 else 'No'
                    })
        
        posthoc_df = pd.DataFrame(posthoc_results)
        st.dataframe(posthoc_df, use_container_width=True)

# Sidebar Controls
with st.sidebar:
    st.header("🔧 Analysis Controls")
    
    # Study parameters
    st.subheader("Study Parameters")
    alpha_level = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
    
    # Filter controls
    st.subheader("Data Filters")
    selected_groups = st.multiselect(
        "Select Treatment Groups",
        options=experimental_df['Group'].unique(),
        default=experimental_df['Group'].unique()
    )
    
    age_range = st.slider(
        "Age Range",
        int(experimental_df['Age'].min()),
        int(experimental_df['Age'].max()),
        (int(experimental_df['Age'].min()), int(experimental_df['Age'].max()))
    )
    
    gender_filter = st.multiselect(
        "Gender",
        options=experimental_df['Gender'].unique(),
        default=experimental_df['Gender'].unique()
    )
    
    # Gene expression filters
    st.subheader("Gene Expression Filters")
    pvalue_threshold = st.slider("P-value Threshold", 0.001, 0.1, 0.05, 0.001)
    fc_threshold = st.slider("Fold Change Threshold", 0.5, 3.0, 1.0, 0.1)
    
    # Export options
    st.subheader("📥 Export Options")
    if st.button("Download Statistical Report"):
        st.success("Statistical report would be generated here!")
    
    if st.button("Export Raw Data"):
        st.success("Raw data would be exported here!")
    
    if st.button("Generate Figure Package"):
        st.success("High-resolution figures would be packaged here!")

# Footer with study metadata
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p><strong>Study Protocol ID:</strong> SCI-2024-001 | <strong>IRB Approval:</strong> IRB-2024-123 | <strong>Last Updated:</strong> {}</p>
    <p><em>This dashboard presents preliminary results for research purposes only. Clinical decisions should not be based solely on this data.</em></p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)