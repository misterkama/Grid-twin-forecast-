# 🇰🇪 Kenya National Grid Twin

A real-time AI-powered dashboard that forecasts electricity demand for all 8 regions of Kenya using LSTM neural networks and live weather forecasts from Open-Meteo.

## Features
- 🌦️ Live weather data + 24-hour forecast for 8 cities
- 🧠 LSTM deep learning model trained on 18 months of EPRA data
- 📊 Interactive 24-hour demand forecast charts
- 🗺️ Regional breakdown: Nairobi, Coast, North Eastern, Central Rift, North Rift, Mt Kenya, West Kenya, South Nyanza

## How to run locally
```bash
pip install -r requirements.txt
streamlit run national_grid_dashboard_forecast.py