# Use the prepared base image
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set working directory inside container
WORKDIR /app

# Copy your script into the container
COPY starter.py .

# Install any dependencies needed by your script
RUN pip install pandas pyarrow scikit-learn==1.5.0

# Set default command to run the script (can be overridden)
CMD ["python", "starter.py"]
