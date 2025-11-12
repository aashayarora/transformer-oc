FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y wget git python3 python3-pip

RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

RUN pip install scikit-learn pytorch-lightning tensorboard mplhep uproot pandas
RUN pip install git+https://github.com/jkiesele/FastGraphCompute

RUN sed -i \
    's/return torch.cat(\[torch.tensor(\[0\], dtype=torch.int64), change_indices + 1, torch.tensor(\[len(batch)\], dtype=torch.int64)\]).to(torch.int64)/return torch.cat(\[torch.tensor(\[0\], dtype=torch.int64, device=batch.device), change_indices + 1, torch.tensor(\[len(batch)\], dtype=torch.int64, device=batch.device)\]).to(torch.int64)/' \
    /usr/local/lib/python3.10/dist-packages/fastgraphcompute/torch_geometric_interface.py

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
