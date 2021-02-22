import torch.nn as nn
import torch.nn.functional as F
import torch

# class SpecialSpmmFunction(torch.autograd.Function):
#     """Special function for only sparse region backpropataion layer."""
#
#     @staticmethod
#     def forward(ctx, indices, values, shape, b):
#         assert indices.requires_grad == False
#         a = torch.sparse_coo_tensor(indices, values, shape)
#         ctx.save_for_backward(a, b)
#         ctx.N = shape[0]
#         return torch.matmul(a, b)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         a, b = ctx.saved_tensors
#         grad_values = grad_b = None
#         if ctx.needs_input_grad[1]:
#             grad_a_dense = grad_output.matmul(b.t())
#             edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
#             grad_values = grad_a_dense.view(-1)[edge_idx]
#         if ctx.needs_input_grad[3]:
#             grad_b = a.t().matmul(grad_output)
#         return None, grad_values, None, grad_b
#
#
# class SpecialSpmm(nn.Module):
#     def forward(self, indices, values, shape, b):
#         return SpecialSpmmFunction.apply(indices, values, shape, b)
#
#
# class SpGraphAttentionLayer(nn.Module):
#     """
#     Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(SpGraphAttentionLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         # W: F x F'
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W)
#
#         # a: 2F'
#         self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
#         nn.init.xavier_uniform_(self.a)
#
#         self.dropout = nn.Dropout(p=dropout)
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#         self.special_spmm = SpecialSpmm()
#
#     def forward(self, input, adj):
#
#         # input: N x F
#         dv = 'cuda' if input.is_cuda else 'cpu'
#
#         N = input.size()[0]
#         # edge: |E| x 2
#         edge = adj.nonzero().t()
#
#         h = torch.mm(input, self.W)
#         # h: N x F'
#         # assert not torch.isnan(h).any()
#
#         # Self-attention on the nodes - Shared attention mechanism
#         # Concatenate the representations of the nodes based on the neighborhood
#         #edge_h:
#         edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
#         # edge: 2*D x E
#
#         edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
#         # assert not torch.isnan(edge_e).any()
#         # edge_e: E
#
#         # Denominator of the attention mechanism
#         e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
#         # e_rowsum: N x 1
#         # assert not torch.isnan(e_rowsum).any()
#
#         edge_e = self.dropout(edge_e)
#         # edge_e: E
#
#         h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
#         # assert not torch.isnan(h_prime).any()
#         # h_prime: N x out
#
#         h_prime = h_prime.div(e_rowsum)
#         # h_prime: N x out
#         # assert not torch.isnan(h_prime).any()
#
#         if self.concat:
#             # if this layer is not last layer,
#             return F.elu(h_prime)
#         else:
#             # if this layer is last layer,
#             return h_prime
#
# class SpGAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout = 0.1, alpha = 0.2, nheads = 1):
#         """Sparse version of GAT."""
#         super(SpGAT, self).__init__()
#         self.dropout = dropout
#
#         self.attentions = [SpGraphAttentionLayer(nfeat,
#                                                  nhid,
#                                                  dropout=dropout,
#                                                  alpha=alpha,
#                                                  concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         self.out_att = SpGraphAttentionLayer(nhid * nheads,
#                                              nclass,
#                                              dropout=dropout,
#                                              alpha=alpha,
#                                              concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.out_att(x, adj))
#         return x


class GAT(nn.Module):
    def __init__(self, in_features, n_hidden, n_out, dropout, alpha, n_heads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GATLayer(in_features, n_hidden, dropout, alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(n_hidden * n_heads, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x,adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.out_att(x, adj))
        return x

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        # in_features denoted by F
        self.in_features = in_features
        # out_features denoted by F'
        self.out_features = out_features

        self.alpha = alpha
        self.concat = concat
        #\mathbf{W} \in \mathbb{R}^{F \times F'}
        self.W  = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W)


        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        #\mathbf{h} \in \mathbb{R}^{N \times F}
        #\mathbf{Wh} \in \mathbb{R}^{N \times F'}
        Wh = torch.mm(h, self.W)

        edges = adj._indices()
        attention_input = torch.cat((Wh[edges[0,:],:], Wh[edges[1,:],:]), dim=1)
        e = self.leakyrelu(torch.mm(attention_input, self.a)).squeeze()

        # coefficients = torch.zeros(size=(N,N))

        coefficients = torch.sparse_coo_tensor(edges, e, adj.shape, requires_grad=False)
        attention = torch.sparse.softmax(coefficients, dim=1).to_dense()

        # a_input = self.__prepare_attention_input(Wh)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        #
        #
        # zero_vec = -9e15 * torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        # attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __prepare_attention_input(self, Wh):
        #Number of nodes
        N = Wh.size()[0]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N,1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)



