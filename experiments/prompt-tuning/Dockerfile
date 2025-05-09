# Base image with Python
FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/unsloth_env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install system packages
RUN apt-get update && \
    apt-get install -y unzip wget git openjdk-8-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Java 8 as default
RUN update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java && \
    java -version

# Set up working directory
WORKDIR /workspace

# Create virtual environment
RUN python -m venv $VIRTUAL_ENV

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install ipykernel kaggle

# Add kernel to Jupyter
RUN python -m ipykernel install --user --name=unsloth_env --display-name "Python (unsloth_env)"

# Add Kaggle API key
COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

# Download and unzip Kaggle dataset
RUN kaggle datasets download -d ushariranasinghe/car-caption-dataset && \
    unzip car-caption-dataset.zip

# Clone and install pycocoevalcap
RUN git clone https://github.com/salaniz/pycocoevalcap.git && \
    cd pycocoevalcap && pip install .

# Setup SPICE (CoreNLP)
RUN mkdir -p /workspace/spice-1.0 && \
    cd /workspace/spice-1.0 && \
    wget http://nlp.stanford.edu/software/stanford-corenlp-3.6.0.zip && \
    unzip stanford-corenlp-3.6.0.zip

# Default command
CMD [ "bash" ]
