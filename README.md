# Data Visualization Web App with OpenAI and Streamlit

This repository contains a Python script that leverages the OpenAI API and Streamlit to create an interactive web application for generating data visualizations based on user queries. The app allows users to choose a dataset, select a machine learning model, and input a query to generate visualizations.

## Features

- Choose from available machine learning models (ChatGPT-4 and ChatGPT-3.5 Turbo).
- Upload your own CSV datasets for visualization.
- Enter a query to define the visualization requirements.
- Interactive visualizations generated using matplotlib.
- Display of selected datasets in separate tabs.
- Error handling for code execution issues.

## Getting Started

1. Clone this repository to your local machine.
2. Install the required Python packages: `pip install pandas openai streamlit`.
3. Get your OpenAI API key and replace `YOUR_OPENAI_API_KEY` in the script.
4. Run the script: `streamlit run visualization_app.py`.
5. Open your web browser and access the local Streamlit app.

## Usage

1. Choose a dataset from the provided options or upload your own CSV file.
2. Select one or more machine learning models.
3. Enter a query in the text area to define your visualization.
4. Click the "Submit" button to generate visualizations.
5. Visualizations will appear on the screen based on the chosen models.
