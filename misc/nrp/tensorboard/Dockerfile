FROM python:3.11-slim

# Install tensorboard only
RUN pip install --no-cache-dir tensorboard

# Expose tensorboard port
EXPOSE 6006

# Set working directory
WORKDIR /workspace

# Run tensorboard
ENTRYPOINT ["tensorboard"]
CMD ["--logdir=/data/output", "--host=0.0.0.0", "--port=6006", "--reload_interval=30"]
