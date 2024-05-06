# Use an official Python runtime as a base image
FROM python:3.12-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    libhdf5-dev \
    gcc \              
    g++ \            
    build-essential    

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP server.py
ENV FLASK_RUN_HOST 0.0.0.0

# Run Flask when the container launches
CMD ["flask", "run"]
