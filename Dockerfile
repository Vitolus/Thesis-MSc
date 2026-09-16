FROM continuumio/miniconda3:latest
LABEL authors="Vito"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \

COPY env_docker.yml /app/env_docker.yml \

RUN conda env create -f /app/env_docker.yml

SHELL ["conda", "run", "-n", "pyqwen", "/bin/bash", "-c"]

COPY . /app

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pytorch-audio", "python"]

CMD ["orchestrator.py"]