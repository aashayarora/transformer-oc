FROM python:3.11-slim

# Install tensorboard and nginx
RUN apt-get update && \
    apt-get install -y nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir tensorboard

# Configure nginx to serve PNG files
RUN echo 'server {\n\
    listen 8080;\n\
    server_name _;\n\
    \n\
    location / {\n\
        root /data/output;\n\
        autoindex on;\n\
        autoindex_exact_size off;\n\
        autoindex_localtime on;\n\
    }\n\
    \n\
    location ~ \.png$ {\n\
        root /data/output;\n\
        add_header Content-Type image/png;\n\
    }\n\
}' > /etc/nginx/sites-available/default && \
    echo 'daemon off;' >> /etc/nginx/nginx.conf

# Expose tensorboard and nginx ports
EXPOSE 6006 8080

# Set working directory
WORKDIR /workspace

# Create startup script
RUN echo '#!/bin/bash\n\
nginx &\n\
tensorboard "$@"' > /start.sh && chmod +x /start.sh

# Run both services
ENTRYPOINT ["/start.sh"]
CMD ["--logdir=/data/output", "--host=0.0.0.0", "--port=6006", "--reload_interval=30"]
