# Use python3.12 as base image
FROM python:3.12-slim

# Containers work directory
WORKDIR /app

# CPU-only PyTorch: there's no GPU inside Docker on a Mac
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy project requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project files
COPY . .

# Open port 8000 and start program
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]