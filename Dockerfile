FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Set working directory
WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your code
COPY train.py .

CMD ["python", "train.py"]
