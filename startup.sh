#!/bin/bash

# Go to the app folder
cd /home/site/wwwroot

# Upgrade pip
python3 -m pip install --upgrade pip

# Force install everything
python3 -m pip install -r requirements.txt

# Run Streamlit
python3 -m streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS false
