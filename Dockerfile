FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y libsndfile1

COPY requirements_deploy.txt requirements.txt
# Ensure we have the CPU-only version of torch for a smaller image
RUN pip install --no-cache-dir -r requirements.txt

# Copy the standalone app and the model artifacts
COPY main.py .
COPY ./best_model ./best_model

# Expose the port Gradio will run on
EXPOSE 7860

# The command to start the Gradio application
CMD ["python", "main.py"]
              