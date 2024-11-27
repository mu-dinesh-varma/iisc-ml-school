import functools
import json
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms
from torch.utils.data import DataLoader, Dataset


@dataclass
class Data:
    """
    Class to contain graph attributes.

    N and M are the number of nodes and edges in the graph, respectively.

    Parameters
    ----------
    node_feat : Tensor
        The node features as a (N, n_node_feats) Tensor.
    edge_feat : Tensor
        The edge features as a (M, n_edge_feats) Tensor.
    edge_src : LongTensor
        The index of the central node for each edge.
    edge_dst : LongTensor
        The index of the destination node for each edge.
    target : Tensor
        The target property to learn.
    atoms : Atoms
        An ase atoms object.
    """

    node_feat: torch.Tensor
    edge_feat: torch.Tensor
    edge_src: torch.LongTensor
    edge_dst: torch.LongTensor
    target: torch.Tensor
    atoms: Atoms


@dataclass
class Batch:
    """
    Class to contain batched graph attributes.

    N and M are the number of nodes and edges across all batched graphs,
    respectively.

    G is the number of graphs in the batch.

    Parameters
    ----------
    node_feat : Tensor
        The node features as a (N, n_node_feats) Tensor.
    edge_feat : Tensor
        The edge features as a (M, n_edge_feats) Tensor.
    edge_src : LongTensor
        The index of the central node for each edge.
    edge_dst : LongTensor
        The index of the destination node for each edge.
    target : Tensor
        The target property to learn, as a (G, 1) Tensor.
    batch : LongTensor
        The graph to which each node belongs, as a (N, ) Tensor.
    """

    node_feat: torch.Tensor
    edge_feat: torch.Tensor
    edge_src: torch.LongTensor
    edge_dst: torch.LongTensor
    target: torch.Tensor
    batch: torch.LongTensor

    def to(self, device, non_blocking=False):
        for k, v in self.__dict__.items():
            self.__dict__[k] = v.to(device=device, non_blocking=non_blocking)


def collate_fn(dataset):
    """
    Collate a list of Data objects and return a Batch.

    Parameters
    ----------

    dataset : MaterialsDataset
        The dataset to batch.

    Returns
    -------
    Batch
        A batched dataset.
    """
    batch = Batch([], [], [], [], [], [])
    base_idx = 0
    for i, data in enumerate(dataset):
        batch.node_feat.append(data.node_feat)
        batch.edge_feat.append(data.edge_feat)
        batch.edge_src.append(data.edge_src + base_idx)
        batch.edge_dst.append(data.edge_dst + base_idx)
        batch.target.append(data.target)
        batch.batch.extend([i] * len(data.node_feat))
        base_idx += len(data.node_feat)
    return Batch(
        node_feat=torch.cat(batch.node_feat),
        edge_feat=torch.cat(batch.edge_feat),
        edge_src=torch.cat(batch.edge_src),
        edge_dst=torch.cat(batch.edge_dst),
        batch=torch.LongTensor(batch.batch),
        target=torch.stack(batch.target),
    )


class MaterialsDataset(Dataset):
    
    def __init__(self, filename, cutoff=4, num_gaussians=40):
        """
        A dataset of materials properties.
    
        Parameters
        ----------
        filename : str
            The path to the dataset.
        cutoff : float
            The cutoff radius for searching for neighbors.
        num_gaussians : float
            The number of gaussian functions used in the edge
            embedding expansion.
        """

        with open(filename) as f:
            self.data = json.load(f)
        self.cutoff = cutoff
        self.num_gaussians = num_gaussians

        self.onehot = {}
        for i in range(119):
            self.onehot[i] = [0] * 119
            self.onehot[i][i - 1] = 1

        for entry in self.data:
            atoms = Atoms(
                positions=entry["positions"],
                cell=entry["cell"],
                numbers=entry["numbers"],
                pbc=[True, True, True],
            )
            edge_src, edge_dst, edge_len = neighbor_list(
                "ijd", atoms, cutoff=self.cutoff, self_interaction=False
            )
            entry.update(
                {
                    "atoms": atoms,
                    "edge_src": edge_src,
                    "edge_dst": edge_dst,
                    "edge_len": edge_len,
                }
            )

    def __len__(self):
        return len(self.data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        entry = self.data[idx]

        # one hot encode element type
        node_feat = torch.Tensor(np.vstack([self.onehot[i] for i in entry["numbers"]]))

        # one hot encode bond length
        filter = np.linspace(0, self.cutoff, self.num_gaussians)
        edge_feat = np.exp(
            -((entry["edge_len"][..., None] - filter) ** 2) / (filter[1] - filter[0]) ** 2
        )

        return Data(
            node_feat=torch.Tensor(node_feat),
            edge_feat=torch.Tensor(edge_feat),
            edge_src=torch.LongTensor(entry["edge_src"]),
            edge_dst=torch.LongTensor(entry["edge_dst"]),
            target=torch.Tensor([entry["y"]]),
            atoms=entry["atoms"],
        )


def plot_sample(sample):
    """Plot the crystal structure and graph of a sample."""
    fig, ax = plt.subplots(1, 2)

    palette = ["#2876B2", "#F39957", "#67C7C2", "#C86646"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cmap", [palette[k] for k in [0, 2, 1]]
    )
    norm = plt.Normalize(vmin=0, vmax=len(sample.atoms) - 1)
    z = dict(zip(sample.atoms.numbers, range(len(sample.atoms))))
    color = [
        mpl.colors.to_hex(k) for k in cmap(norm([z[j] for j in sample.atoms.numbers]))
    ]
    plot_atoms(sample.atoms, ax[0], radii=0.25, colors=color)
    ax[0].set(xlabel=r"$x_1\ (\AA)$", ylabel=r"$x_2\ (\AA)$", title="Crystal structure")

    graph = nx.Graph()
    graph.add_edges_from(zip(sample.edge_src.tolist(), sample.edge_dst.tolist()))
    nx.draw_networkx(
        graph,
        ax=ax[1],
        labels=dict(zip(range(len(sample.atoms.symbols)), list(sample.atoms.symbols))),
        node_size=500,
        node_color=color,
        edge_color="gray",
    )
    ax[1].set(aspect="equal", title="Crystal graph")
    ax[1].axis("off")
