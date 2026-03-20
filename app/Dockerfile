# 1. Use Python 3.10 Slim (Lightweight)
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /code

# 3. Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install them
COPY requirements.txt .

# FIX: Added '--default-timeout=1000' to prevent crash on slow internet
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 5. Copy the entire project into the container
COPY . .

# 6. Expose the port
EXPOSE 8000

# 7. Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]