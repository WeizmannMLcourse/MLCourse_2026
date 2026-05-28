import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool


### Fully-connected network ###
class FCN(nn.Module):

    def __init__(self, N_input, N_hidden, N_output):
        super().__init__()

        assert len(N_hidden) > 0, "Pass list of hidden layer sizes for N_hidden"

        self.node_network = nn.Sequential(
            nn.Linear(N_input, N_hidden[0]),
            nn.ReLU(),
        )

        if len(N_hidden) > 1:
            for i in range(1, len(N_hidden)):
                self.node_network.append(nn.Linear(N_hidden[i - 1], N_hidden[i]))
                self.node_network.append(nn.ReLU())


        self.node_network.append(nn.Linear(N_hidden[-1], N_output))

    def forward(self, x):

        return self.node_network(x)
    


### Node update network ###
class MessagePassingBlock(MessagePassing):

    def __init__(self, N_hidden):
        super().__init__(aggr='mean')

        self.net = FCN(N_hidden*2, [N_hidden*3, N_hidden*2, N_hidden*2], N_hidden)

    def message(self, h_j):
        return h_j

    def update(self, aggregate, h):
        rep_cat_aggregate = torch.cat([h, aggregate], dim=-1)
        updated_rep = self.net(rep_cat_aggregate)
        return updated_rep

    def forward(self, h, edge_index):
        return self.propagate(edge_index, h=h)


### Message-passing neural network ###
class MoleculeMPNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.z_unique = [1, 6, 7, 8, 9]
        self.N_input = 3 + len(self.z_unique)
        self.N_hidden = 64
        self.N_output = 1
        self.N_message_passing_blocks = 4

        self.node_embedding = FCN(self.N_input, 
                                 [self.N_hidden, self.N_hidden], 
                                  self.N_hidden)

        self.mp_networks  = nn.ModuleList(
            [ MessagePassingBlock(self.N_hidden) for _ in range(self.N_message_passing_blocks)]
        )

        self.pred_networks = FCN(self.N_hidden, 
                                 [self.N_hidden*2, self.N_hidden*3, self.N_hidden*2, self.N_hidden], 
                                 self.N_output)

    def forward(self, g):

        ### Extract atomic number...
        z = g.z
        ### ... make compact version based on unique values ...
        z_compact = torch.zeros_like(z)
        for i, z_val in enumerate(self.z_unique):
            z_compact[z == z_val] = i
        ### ... and one-hot encode it
        z_one_hot = F.one_hot(z_compact, num_classes=len(self.z_unique)).to(torch.float)

        ### Extract position and atomic number
        feats = torch.cat([g.pos, z_one_hot], dim=-1)

        ### Embed the node features
        h = self.node_embedding(feats)

        ### Message-passing blocks
        for i in range(self.N_message_passing_blocks):
            h = self.mp_networks[i](h, g.edge_index)

        ### This allows to forward-pass a single graph as well as a batched graph
        batch = getattr(g, 'batch', None)
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)

        ### Global representation
        global_rep = global_add_pool(h, batch)

        ### Prediction
        pred = self.pred_networks(global_rep)

        return pred