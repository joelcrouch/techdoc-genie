FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies globally
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for API
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# FROM python:3.10-slim

# WORKDIR /app

# # Install system dependencies
# # These might still be needed for certain Python packages (e.g., build-essential)
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Create a virtual environment
# ENV VIRTUAL_ENV=/app/techdoc-genie-venv
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# RUN python3 -m venv $VIRTUAL_ENV

# # Copy requirements file and install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Expose port for API
# EXPOSE 8000

# # The CMD assumes src/api/main.py exists and has an 'app' instance.
# # It also assumes you want to run the FastAPI app directly.
# CMD ["/app/techdoc-genie-venv/bin/uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]