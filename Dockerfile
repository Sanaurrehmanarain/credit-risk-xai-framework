# Use Python 3.13.5 slim for maximum compatibility with ML libraries
FROM python:3.13.5-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependency required for LightGBM/XGBoost to run correctly
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure (data, notebooks, models, plots) into the container
COPY . .

# Expose ports for both Jupyter Lab (8888) and Streamlit (8501)
EXPOSE 8888 8501

# Default Command: Launch Jupyter Lab so reviewers can see the notebooks
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]