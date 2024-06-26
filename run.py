import argparse
import ctypes
from ctypes import c_size_t, c_bool, c_uint16, c_int, POINTER
import json
import math
import os
import time

import numpy as np
from numpy.linalg import eigh
from scipy.sparse.csgraph import laplacian
import torch


GRAPH_SIZES = {
    "small-graph": 1357,
    "medium-graph": 1399,
    "large-graph": 2426,
}


# Initialise custom CUDA kerrnel to quickly evaluate solutions
libeval = ctypes.CDLL('./libeval.so', mode=ctypes.RTLD_GLOBAL)
evaluate = libeval.evaluate
evaluate.argtypes = [
    POINTER(c_bool),    # adjs
    POINTER(c_uint16),  # perms
    POINTER(c_uint16),  # degrees
    POINTER(c_int),     # fitnesses
    c_size_t,           # B
    c_size_t,           # N
]


def main():
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=set(GRAPH_SIZES.keys()), default="small-graph")
    parser.add_argument("--eigenvectors", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_generations", type=int, default=100_000)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()

    B = args.batch_size
    N = GRAPH_SIZES[args.graph]
    E = args.eigenvectors + 5

    # Initialise representations of the graph
    adj = init_adj(N, args.graph).astype(np.bool_)
    adjs = np.repeat(adj[None, :, :], B, 0)

    nodes = init_node_features(adj, args.eigenvectors)
    nodes = torch.from_numpy(nodes).cuda()

    # Pre-allocate tensors on the GPU
    population = np.random.normal(0.0, 0.3, (B, E)).astype(np.float32)
    elites = torch.empty((N, E), dtype=torch.float32).cuda()

    bkup_adjs = torch.from_numpy(adjs).cuda()
    adjs = torch.from_numpy(adjs).cuda()
    population = torch.from_numpy(population).cuda()

    perms = torch.empty((B, N), dtype=torch.uint16).cuda()
    degrees = torch.empty((B, N), dtype=torch.uint32).cuda()
    fitnesses = torch.full((B, N), 999999, dtype=torch.int).cuda()
    elite_fitnesses = torch.full((N, ), 999999, dtype=torch.int).cuda()

    elite_range = torch.ones((N, ), dtype=torch.float32, device="cuda") / B
    ts = torch.arange(0, N, math.ceil(N / 20)).cuda()

    logits = torch.empty((B, N), dtype=torch.float32).cuda()

    # Run multiple generations of neuro-evolution
    for generation in range(args.max_generations):
        start = time.perf_counter()
        logits[:] = population @ nodes
        perms[:] = logits.argsort(axis=1).to(torch.uint16)
        adjs[:, :, :] = bkup_adjs[:, :, :]
        evaluate(
            ctypes.cast(adjs.data_ptr(), POINTER(c_bool)),
            ctypes.cast(perms.data_ptr(), POINTER(c_uint16)),
            ctypes.cast(degrees.data_ptr(), POINTER(c_uint16)),
            ctypes.cast(fitnesses.data_ptr(), POINTER(c_int)),
            B, 
            N,
        )

        # Save the best from the population
        best = fitnesses.min(axis=0)
        better = best.values <= elite_fitnesses 
        elites[better] = population[best.indices][better]
        elite_fitnesses[better] = best.values[better]
        
        # Create next population
        idx = torch.multinomial(elite_range, B, replacement=N < B)
        population[:] = elites[idx]
        mask = torch.rand((B, E), device="cuda") > 0.5
        population += torch.normal(0.0, 0.3, (B, E), device="cuda") * mask

        if generation % args.log_every == 0:
            # Log stats
            runtime = time.perf_counter() - start
            degrees = elite_fitnesses[ts]
            hvi = calculate_hvi(ts, degrees, N)
            print(f"{generation} | {runtime / args.log_every:3.2f} | {(B * args.log_every)/runtime:3.2f} | {best.values.sum()} | {elite_fitnesses.sum()} | {hvi}")

        if generation % args.checkpoint_every == 0:
            # Checkpoint progress

            # Submission
            degrees = elite_fitnesses[ts]
            hvi = calculate_hvi(ts, degrees, N)
            submission = create_submission(elites, nodes, ts, args.graph)
            submission_path = f"submissions/{args.graph}/{hvi}.json"
            if not os.path.exists(submission_path):
                with open(submission_path, "w") as f:
                    json.dump(submission, f)
            
            # State
            checkpoint_path = f"checkpoints/{args.graph}/{elite_fitnesses.sum()}.pt"
            if not os.path.exists(checkpoint_path):
                torch.save(
                    {
                        "args": vars(args),
                        "nodes": nodes,
                        "elites": elites,
                        "elite_fitnesses": elite_fitnesses,
                    },
                    checkpoint_path,
                )


def create_submission(elites, nodes, ts, graph):
    """Create a submission based on the elites."""
    logits = elites @ nodes
    perms = logits.argsort(axis=1)
    return {
        "challenge": "spoc-3-torso-decompositions",
        "problem": graph,
        "decisionVector": [perm.tolist() + [t.item()] for perm, t in zip(perms[ts].cpu(), ts.cpu())],
    }


def calculate_hvi(ts, degrees, N):
    """Calculate the Hyper Volume Indicator used to score a submission."""
    area = 0
    for i in range(len(ts)):
        degree = degrees[i].item()
        previous_degree = degrees[i - 1].item() if i != 0 else N
        x = ts[i].item()
        y = degree
        dx = N - x
        dy = previous_degree - y
        contribution = dx * dy
        area += contribution
    return -area


def init_adj(N, graph):
    """Initialise an adjacency matrix based on the edges."""
    adj = np.zeros((N, N), dtype=np.bool_)
    with open(f"data/{graph}.gr", "r") as f:
        for line in f.readlines():
            src, dst = line.split(" ")
            src = int(src.strip())
            dst = int(dst.strip())
            adj[src, dst] = True
            adj[dst, src] = True
    return adj


def init_node_features(adj, eigenvectors):
    """Initialise node features based on the graph topology."""
    N = adj.shape[0]
    degree_profile = 5
    features = eigenvectors + degree_profile
    nodes = np.zeros((N, features), dtype=np.float32)

    # Degree based features based on pytorch geometrics 'LocalDegreeProfile' https://pytorch-geometric.readthedocs.io/en/2.5.0/_modules/torch_geometric/transforms/local_degree_profile.html#LocalDegreeProfile
    for i in range(N):
        nodes[i, 0] = adj[i].sum()
    for i in range(N):
        nodes[i, 1] = nodes[adj[i], 0].min()
        nodes[i, 2] = nodes[adj[i], 0].max()
        nodes[i, 3] = nodes[adj[i], 0].mean()
        nodes[i, 4] = nodes[adj[i], 0].std()

    # Eigenvectors of the laplacien based on pytorch geometrics 'AddLaplacianEigenvectorPE' https://pytorch-geometric.readthedocs.io/en/2.5.0/_modules/torch_geometric/transforms/add_positional_encoding.html#AddLaplacianEigenvectorPE
    lap = laplacian(adj.astype(np.int8), normed=True)
    eig_vals, eig_vecs = eigh(lap)
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = eig_vecs[:, 1: eigenvectors + 1]
    nodes[:, degree_profile:] = pe

    # Normalisation (mean 0.0, stdev 1.0)
    means = np.empty((features, ), dtype=np.float32)
    stds = np.empty((features, ), dtype=np.float32)
    for i in range(features):
        means[i] = nodes[:, i].mean()
        stds[i] = nodes[:, i].std()
    nodes = (nodes - means) / stds

    return nodes.T


if __name__ == "__main__":
    main()
