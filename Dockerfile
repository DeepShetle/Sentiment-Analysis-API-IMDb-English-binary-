# Text file to define the environment for the application

FROM python:3.12-slim
# Define the base image to build the app

WORKDIR /app    
# Default working directory is /app

# Install dependencies first, separate from COPY code —
# leverage Docker layer cache: if only code is modified, not requirements.txt,
# the pip install step won't run again (much faster build)
COPY requirements.txt .
# Copy requirements file to the current directory "." (which is /app)
RUN pip install --no-cache-dir -r requirements.txt
# Download and install, no cache to optimize image size

# Copy necessary code for runtime — DO NOT copy notebooks/, data/raw/ (not needed for serving)
COPY app/ ./app/
COPY src/ ./src/
COPY teencode_dict.json .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]