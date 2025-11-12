FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y wget git python3 python3-pip

RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

RUN pip install scikit-learn pytorch-lightning tensorboard mplhep uproot pandas
RUN pip install git+https://github.com/jkiesele/FastGraphCompute

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
