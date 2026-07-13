FROM python:3.12 AS main

WORKDIR /app

# Install pandoc and netcat
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    pandoc \
    netcat-openbsd \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download standard NLTK data, to prevent unstructured from downloading packages at runtime.
# Use the Python API with explicit package IDs (rather than `python -m nltk.downloader`),
# which never falls back to the interactive prompt that EOFs during a non-interactive
# build; unknown IDs are tolerated so the build stays version-agnostic across nltk releases.
RUN python -c "import nltk; [nltk.download(p, download_dir='/app/nltk_data') for p in ['punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']]"
ENV NLTK_DATA=/app/nltk_data

# Disable Unstructured analytics
ENV SCARF_NO_ANALYTICS=true

COPY . .

CMD ["python", "main.py"]
