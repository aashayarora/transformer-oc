""" Utility functions for smyrf """
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from einops import rearrange

from typing import List


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Params:
        values: (1 or n_hashes, batch, seqlen, dim)
        indices: (n_hashes, batch, seqlen)
    Return:
        (n_hashes, batch, seqlen, dim)
    """
    last_dim = values.shape[-1]
    indices_expanded = rearrange(indices, "... -> ... 1").expand(*indices.shape, last_dim)
    return values.expand(*indices_expanded.shape[:-2], *values.shape[-2:]).gather(-2, indices_expanded)


@torch.no_grad()
def lsh_mapping(e2lsh, queries, keys):
    queries_hashed = e2lsh(queries)
    keys_hashed = e2lsh(keys)
    max_hash_shift = torch.max(queries_hashed.max(-1, keepdim=True).values, keys_hashed.max(-1, keepdim=True).values)
    min_hash_shift = torch.min(queries_hashed.min(-1, keepdim=True).values, keys_hashed.min(-1, keepdim=True).values)
    hash_shift = max_hash_shift - min_hash_shift
    return queries_hashed, keys_hashed, hash_shift


def sort_to_buckets(x, perm, bucketsz):
    return rearrange(
        batched_index_select(rearrange(x, "b s d -> 1 b s d"), perm),
        "h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d",
        bucketsz=bucketsz,
    )


def unsort_from_buckets(s_x, perm_inverse):
    b_x = rearrange(s_x, "h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d")
    return batched_index_select(b_x, perm_inverse)


def qkv_res(s_query, s_key, s_value):
    q_sq_05 = -0.5 * (s_query**2).sum(dim=-1, keepdim=True)
    k_sq_05 = -0.5 * (s_key**2).sum(dim=-1, keepdim=True)

    clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key)
    clustered_dists = (clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)).clamp(max=0.0).exp()

    denom = clustered_dists.sum(dim=-1, keepdim=True) + (1e-20)
    qk = clustered_dists

    so = torch.einsum("...ij,...jd->...id", qk, s_value)
    return denom, so


def prep_qk(query, key, w, coords):
    qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
    new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)

    sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim)[None] * coords[:, None]
    q_hat = torch.cat([query, sqrt_w_r], dim=-1)
    k_hat = torch.cat([key, sqrt_w_r], dim=-1)
    return q_hat, k_hat


@torch.no_grad()
def get_geo_shift(regions_h: List[List[int]], hash_shift, region_indices, num_or_hashes):
    region_indices_eta, region_indices_phi = region_indices

    q_hash_shift_eta = region_indices_eta * hash_shift
    k_hash_shift_eta = region_indices_eta * hash_shift

    q_hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    k_hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    res = torch.stack([q_hash_shift_phi + q_hash_shift_eta, k_hash_shift_phi + k_hash_shift_eta], dim=0)
    return rearrange(res, "a (c h) n -> a c h n", c=num_or_hashes)

def quantile_partition(sorted_indices, num_regions):
    total_elements = sorted_indices.shape[-1]
    region_size = torch.ceil(total_elements / num_regions)
    inverse_indices = torch.argsort(sorted_indices, dim=-1)

    base = torch.arange(total_elements, device=sorted_indices.device)[None]
    region_indices = base // region_size + 1
    reassigned_regions = region_indices[:, inverse_indices]
    return reassigned_regions


def get_regions(num_regions, num_or_hashes, num_heads, num_and_hashes=2):
    lb = 2
    ub = 2 * num_regions ** (1 / num_and_hashes) - lb
    regions = []
    for _ in range(num_or_hashes * num_heads):
        region = []
        for _ in range(num_and_hashes):
            a = torch.rand(1).item() * (ub - lb) + lb
            region.append(a)
        regions.append(region)
    regions = torch.tensor(regions)
    regions = (num_regions / regions.prod(dim=1, keepdim=True)) ** (1 / num_and_hashes) * regions

    regions = torch.round(regions * 3) / 3
    return rearrange(regions, "(h c) a -> c a h", h=num_heads)


class E2LSH(nn.Module):
    def __init__(self, n_hashes, n_heads, dim, r=1):
        super(E2LSH, self).__init__()

        self.alpha = nn.Parameter(torch.normal(0, 1, (n_heads, dim, n_hashes)))
        self.alpha.requires_grad = False

    def forward(self, vecs):
        projection = torch.bmm(vecs, self.alpha)
        return projection.permute(2, 0, 1)


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    """
    Params:
        perm: (..., n)
    Return:
        inverse_perm: (..., n)
    """
    # This is simpler but has complexity O(n log n)
    # return torch.argsort(perm, dim=-1)
    # This is more complicated but has complexity O(n)
    arange = torch.arange(perm.shape[-1], device=perm.device).expand_as(perm)
    return torch.empty_like(perm).scatter_(-1, perm, arange)


def bit_shift(base, shift_idx):
    max_base = base.max(dim=1, keepdim=True).values
    num_bits = torch.ceil(torch.log2(max_base + 1)).long()
    return (shift_idx << num_bits) | base


def pad_and_unpad(batch, block_size, region_indices, raw_sizes):
    padded_sizes = ((raw_sizes + block_size - 1) // block_size) * block_size
    pad_sizes = padded_sizes - raw_sizes

    pad_cumsum = padded_sizes.cumsum(0)
    pad_seq = torch.arange(pad_cumsum[-1], device=batch.device)
    unpad_seq = torch.ones(pad_cumsum[-1], device=batch.device).bool()

    sorted_region_indices = region_indices.argsort()
    for i in range(len(raw_sizes)):
        idx_to_fill = pad_cumsum[i] - block_size - pad_sizes[i] + torch.arange(pad_sizes[i], device=batch.device)
        if i >= 1:
            pad_seq[pad_cumsum[i - 1] :] -= pad_sizes[i - 1]
            idx_to_fill -= pad_sizes[:i].sum()
        pad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = sorted_region_indices[idx_to_fill]
        unpad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = False
    return pad_seq, unpad_seq


def prepare_input(x, coords, batch, helper_params):
    kwargs = {}
    regions = rearrange(helper_params["regions"], "c a h -> a (c h)")
    with torch.no_grad():
        block_size, num_heads = helper_params["block_size"], helper_params["num_heads"]
        graph_sizes = batch.bincount()
        graph_size_cumsum = graph_sizes.cumsum(0)
        
        raw_size = x.shape[0]

        region_indices_eta, region_indices_phi = [], []
        for graph_idx in range(len(graph_size_cumsum)):
            start_idx = 0 if graph_idx == 0 else graph_size_cumsum[graph_idx - 1]
            end_idx = graph_size_cumsum[graph_idx]
            sorted_eta_idx = torch.argsort(coords[start_idx:end_idx, 0], dim=-1)
            sorted_phi_idx = torch.argsort(coords[start_idx:end_idx, 1], dim=-1)

            region_indices_eta.append(quantile_partition(sorted_eta_idx, regions[0][:, None]))
            region_indices_phi.append(quantile_partition(sorted_phi_idx, regions[1][:, None]))
        region_indices_eta = torch.cat(region_indices_eta, dim=-1)
        region_indices_phi = torch.cat(region_indices_phi, dim=-1)

        combined_shifts = bit_shift(region_indices_eta.long(), region_indices_phi.long())
        combined_shifts = bit_shift(combined_shifts, batch[None])
        combined_shifts = rearrange(combined_shifts, "(c h) n -> c h n", h=num_heads)

        pad_seq, unpad_seq = pad_and_unpad(batch, block_size, combined_shifts[0, 0], graph_sizes)
        x = x[pad_seq]
        kwargs["combined_shifts"] = combined_shifts[..., pad_seq]
        kwargs["coords"] = coords[pad_seq]
        kwargs["raw_size"] = raw_size
        kwargs["regions_h"] = regions
        kwargs["region_indices"] = (region_indices_eta[..., pad_seq], region_indices_phi[..., pad_seq])
    return x, kwargs, unpad_seq

class HEPTAttention(nn.Module):
    def __init__(self, hash_dim, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)

        self.block_size = kwargs["block_size"]
        self.n_hashes = kwargs["n_hashes"]
        self.num_w_per_dist = kwargs["num_w_per_dist"]
        self.e2lsh = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=hash_dim)

    def forward(self, query, key, value, **kwargs):
        # TODO: support batched inputs
        query = query.view(-1, self.num_heads, self.dim_per_head)
        key = key.view(-1, self.num_heads, self.dim_per_head)
        value = value.view(-1, self.num_heads, self.dim_per_head)

        w = rearrange(
            kwargs["w_rpe"].weight,
            "(h d) (r k) -> h d r k",
            h=self.num_heads,
            d=self.dim_per_head,
            k=self.num_w_per_dist,
        )
        q_hat, k_hat = prep_qk(query, key, w, kwargs["coords"])

        q_hat = rearrange(q_hat, "n h d -> h n d")
        k_hat = rearrange(k_hat, "n h d -> h n d")
        value = rearrange(value, "n h d -> h n d")
        q_hat[:, kwargs["raw_size"] :] = 0.0
        k_hat[:, kwargs["raw_size"] :] = 0.0
        value[:, kwargs["raw_size"] :] = 0.0

        q_hashed, k_hashed, hash_shift = lsh_mapping(self.e2lsh, q_hat, k_hat)
        hash_shift = rearrange(hash_shift, "c h d -> (c h) d")
        q_hashed[..., kwargs["raw_size"] :] = float("inf")
        k_hashed[..., kwargs["raw_size"] :] = float("inf")

        q_shifts, k_shifts = get_geo_shift(kwargs["regions_h"], hash_shift, kwargs["region_indices"], self.n_hashes)

        q_hashed = q_hashed + q_shifts
        k_hashed = k_hashed + k_shifts

        q_positions = q_hashed.argsort(dim=-1)
        k_positions = k_hashed.argsort(dim=-1)

        s_query = sort_to_buckets(q_hat, q_positions, self.block_size)
        s_key = sort_to_buckets(k_hat, k_positions, self.block_size)
        s_value = sort_to_buckets(value, k_positions, self.block_size)

        denom, so = qkv_res(s_query, s_key, s_value)

        q_rev_positions = invert_permutation(q_positions)
        o = unsort_from_buckets(so, q_rev_positions)
        logits = unsort_from_buckets(denom, q_rev_positions)
        out = o.sum(dim=0) / logits.sum(dim=0)
        out = self.out_linear(rearrange(out, "h n d -> n (h d)"))
        return out
    