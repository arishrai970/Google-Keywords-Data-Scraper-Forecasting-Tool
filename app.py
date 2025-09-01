import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from datetime import datetime, timedelta
import time

# Set page configuration
st.set_page_config(
    page_title="Google Keywords Forecaster",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🔍 Google Keywords Data Scraper & Forecaster")
st.markdown("""
This tool helps you analyze and forecast Google search trends across different countries, regions, and languages.
**Note:** This demo uses simulated data as directly scraping Google search data is against their terms of service.
""")

# Sidebar for inputs
st.sidebar.header("Search Parameters")

# Country selection
countries = {
    "United States": "US",
    "United Kingdom": "UK", 
    "Canada": "CA",
    "Australia": "AU",
    "Germany": "DE",
    "France": "FR",
    "India": "IN",
    "Japan": "JP",
    "Brazil": "BR",
    "Mexico": "MX"
}

selected_country = st.sidebar.selectbox(
    "Select Country",
    list(countries.keys())
)

# Region selection based on country
regions = {
    "US": ["California", "Texas", "New York", "Florida", "Illinois", "All"],
    "UK": ["England", "Scotland", "Wales", "Northern Ireland", "All"],
    "CA": ["Ontario", "Quebec", "British Columbia", "Alberta", "All"],
    "AU": ["New South Wales", "Victoria", "Queensland", "Western Australia", "All"],
    "DE": ["Bavaria", "Berlin", "Hamburg", "North Rhine-Westphalia", "All"],
    "FR": ["Île-de-France", "Provence-Alpes-Côte d'Azur", "Auvergne-Rhône-Alpes", "All"],
    "IN": ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "All"],
    "JP": ["Tokyo", "Osaka", "Kyoto", "Hokkaido", "All"],
    "BR": ["São Paulo", "Rio de Janeiro", "Minas Gerais", "Bahia", "All"],
    "MX": ["Mexico City", "Jalisco", "Nuevo León", "Baja California", "All"]
}

selected_region = st.sidebar.selectbox(
    "Select Region/State",
    regions[countries[selected_country]]
)

# Language selection
languages = ["English", "Spanish", "French", "German", "Portuguese", "Hindi", "Japanese", "All"]
selected_language = st.sidebar.selectbox("Select Language", languages)

# Search term input
search_term = st.sidebar.text_input("Enter Search Term", "artificial intelligence")

# Date range selection
date_range = st.sidebar.selectbox(
    "Select Date Range",
    ["Last 7 days", "Last 30 days", "Last 90 days", "Last 6 months", "Last year"]
)

# Forecasting options
forecast_period = st.sidebar.slider("Forecast Period (days)", 7, 90, 30)

# Generate mock data function
def generate_mock_data(keyword, country, region, language, periods=90):
    """Generate mock search volume data"""
    dates = pd.date_range(end=datetime.today(), periods=periods).tolist()
    
    # Base trend based on keyword
    base_trend = np.random.normal(1000, 200, periods)
    
    # Seasonality (weekly pattern)
    day_of_week = [date.weekday() for date in dates]
    seasonality = 200 * np.sin(2 * np.pi * np.array(day_of_week) / 7)
    
    # Country factor
    country_factors = {
        "US": 1.2, "UK": 1.0, "CA": 0.8, "AU": 0.7, 
        "DE": 0.9, "FR": 0.9, "IN": 1.5, "JP": 0.8, "BR": 0.9, "MX": 0.8
    }
    country_factor = country_factors.get(country, 1.0)
    
    # Random noise
    noise = np.random.normal(0, 100, periods)
    
    # Combine all components
    search_volume = base_trend + seasonality + noise
    search_volume = search_volume * country_factor
    search_volume = np.maximum(search_volume, 0)  # Ensure non-negative
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'search_volume': search_volume.astype(int),
        'keyword': keyword,
        'country': country,
        'region': region,
        'language': language
    })
    
    return df

# Forecast function
def forecast_search_volume(df, periods=30):
    """Forecast search volume using linear regression"""
    # Prepare data for modeling
    df = df.copy()
    df['days'] = (df['date'] - df['date'].min()).dt.days
    
    # Features: days and day of week
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Train model
    X = df[['days', 'day_of_week']]
    y = df['search_volume']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future dates
    last_date = df['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, periods+1)]
    
    # Prepare future features
    future_days = [(date - df['date'].min()).days for date in future_dates]
    future_day_of_week = [date.weekday() for date in future_dates]
    
    X_future = pd.DataFrame({
        'days': future_days,
        'day_of_week': future_day_of_week
    })
    
    # Make predictions
    future_volume = model.predict(X_future)
    future_volume = np.maximum(future_volume, 0)  # Ensure non-negative
    
    # Create future DataFrame
    future_df = pd.DataFrame({
        'date': future_dates,
        'search_volume': future_volume.astype(int),
        'keyword': df['keyword'].iloc[0],
        'country': df['country'].iloc[0],
        'region': df['region'].iloc[0],
        'language': df['language'].iloc[0],
        'is_forecast': True
    })
    
    # Add is_forecast flag to historical data
    df['is_forecast'] = False
    
    # Combine historical and forecast data
    result_df = pd.concat([df, future_df], ignore_index=True)
    
    return result_df, model

# Main app
if st.sidebar.button("Analyze Search Term"):
    with st.spinner(f"Generating data and analysis for '{search_term}' in {selected_country}..."):
        # Generate mock data
        data = generate_mock_data(
            search_term, 
            countries[selected_country], 
            selected_region, 
            selected_language
        )
        
        # Perform forecasting
        forecast_data, model = forecast_search_volume(data, forecast_period)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Search Volume", f"{data['search_volume'].iloc[-1]:,}")
        with col2:
            change = ((data['search_volume'].iloc[-1] - data['search_volume'].iloc[-2]) / data['search_volume'].iloc[-2]) * 100
            st.metric("Change from Previous Day", f"{change:.1f}%")
        with col3:
            st.metric("Average Volume", f"{data['search_volume'].mean():.0f}")
        
        # Plot historical and forecast data
        st.subheader(f"Search Volume Trend for '{search_term}'")
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Historical data
        historical = forecast_data[~forecast_data['is_forecast']]
        fig.add_trace(go.Scatter(
            x=historical['date'],
            y=historical['search_volume'],
            mode='lines',
            name='Historical',
            line=dict(color='#1f77b4')
        ))
        
        # Forecast data
        future = forecast_data[forecast_data['is_forecast']]
        fig.add_trace(go.Scatter(
            x=future['date'],
            y=future['search_volume'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', dash='dash')
        ))
        
        # Confidence interval (simplified)
        if len(future) > 0:
            last_historical_value = historical['search_volume'].iloc[-1]
            confidence_upper = future['search_volume'] * 1.2
            confidence_lower = future['search_volume'] * 0.8
            
            fig.add_trace(go.Scatter(
                x=future['date'],
                y=confidence_upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=future['date'],
                y=confidence_lower,
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(width=0),
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Search Volume',
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.subheader("Data Table")
        st.dataframe(forecast_data)
        
        # Download button
        csv = forecast_data.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"search_volume_{search_term}_{selected_country}.csv",
            mime="text/csv"
        )

else:
    # Show instructions when no analysis has been run
    st.info("👈 Enter a search term and configure parameters in the sidebar to begin analysis.")

# Footer
st.markdown("---")
st.markdown(
    """
    **Disclaimer:** This is a demo application using simulated data. 
    Directly scraping Google search data is against their terms of service.
    """
)
