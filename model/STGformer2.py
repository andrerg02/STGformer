import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F

# class SpatialSelfAtt(nn.Module):
#     def __init__(self):
#         self.C = C
#         self.W_Qs = nn.Linear(C, C)
#         self.W_Ks = nn.Linear(C, C)
#         self.W_Vs = nn.Linear(C, C)
#         self.softmax = nn.Sotmax(dim=1) 
#         pass


#     def forward(self, h):
#         #h is T x N x C
#         Q_s = self.W_Qs(h)
#         K_s = self.W_Ks(h)
#         V_s = self.W_Vs(h)

#         A_s = self.softmax(torch.matmul(Q_s, K_s.transpose(1,2))/self.C**0.5)
#         pass

# class TemporalSelfAtt(nn.Module):
#     def __init__(self):
#         self.C = C
#         self.W_Qt = nn.Linear(C, C)
#         self.W_Kt = nn.Linear(C, C)
#         self.W_Vt = nn.Linear(C, C)
#         self.softmax = nn.Sotmax(dim=1) 
#         pass


#     def forward(self, h):
#         #h is T x N x C
#         Q_t = self.W_Qt(h)
#         K_t = self.W_Kt(h)
#         V_t = self.W_Vt(h)

#         A_s = self.softmax(torch.matmul(Q_t.transpose(1,2), K_t)/self.C**0.5)
#         pass

# class SpatialTemporalSelfAtt(nn.Module):
#     def __init__(self):
#         self.C = C
#         self.W_Q = nn.Linear(C, C)
#         self.W_K = nn.Linear(C, C)
#         self.W_V = nn.Linear(C, C)
#         self.softmax = nn.Sotmax(dim=1) 
#         pass


#     def forward(self, h):
#         #h is T x N x C
#         Q = self.W_Q(h)
#         K = self.W_K(h)
#         V = self.W_V(h)

#         A_s = self.softmax(torch.matmul(Q, K.transpose(1,2))/self.C**0.5)
#         A_t = self.softmax(torch.matmul(Q.transpose(1,2), K)/self.C**0.5)
#         pass

class SpatioTemporalLinearizedAttention(nn.Module):
    def __init__(self, model_dim, num_heads=8, kernel=1):
        super(SpatioTemporalLinearizedAttention, self).__init__()
        self.W_Q = nn.Linear(model_dim, model_dim, bias=False)
        self.W_K = nn.Linear(model_dim, model_dim, bias=False)
        self.W_V = nn.Linear(model_dim, model_dim, bias=False)
        self.fc = nn.Linear(2 * model_dim if kernel != 12 else model_dim, model_dim)
    
    def fast_attention(self, Q, K, V, spatial=True):
        N = Q.size(1)
        Q = nn.functional.normalize(Q, dim=-1) # Q é T x N x C
        K = nn.functional.normalize(K, dim=-1)
        ones = torch.ones(N, device=K.device)
        K1 = torch.matmul(K.transpose(1,3), ones)
        D = (Q @ K1[:, None, ...]).sum(dim=-1)
        D += torch.ones_like(D) * N
        QK = Q @ K.transpose(2,3) if spatial else (Q @ K.transpose(2,3)).transpose(2,3)
        QKVNV = QK @ V + N * V
        out = QKVNV / D[... , None]
        return out

    def forward(self, h):
        Q = self.W_Q(h)
        K = self.W_K(h)
        V = self.W_V(h)
        n = Q.size(2)

        A_s = self.fast_attention(Q, K, V, spatial=True)/n
        A_t = self.fast_attention(Q, K, V, spatial=False)/n
        out = self.fc(torch.cat([A_s, A_t], dim=-1))

        return out

class SelfAttention(nn.Module):
    def __init__(self, model_dim, mlp_ratio=2, num_heads=8, dropout=0, mask=False, order=2,):
        super(SelfAttention, self).__init__()
        self.attn = nn.ModuleList([SpatioTemporalLinearizedAttention(model_dim, num_heads, mask) for _ in range(order)])
        self.pws = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(order)])
        for i in range(0, order):
            nn.init.constant_(self.pws[i].weight, 0)
            nn.init.constant_(self.pws[i].bias, 0)
        self.fc = nn.Sequential(nn.Linear(model_dim, model_dim * mlp_ratio), nn.ReLU(), nn.Dropout(dropout), nn.Linear(model_dim * mlp_ratio, model_dim), nn.Dropout(dropout))
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = [1, 0.01, 0.001]

    def forward(self, x, x_graph):
        c = x_glo = x
        for i, z in enumerate(x_graph):
            att_outputs = self.attn[i](z)
            x_glo += att_outputs * self.pws[i](c) * self.scale[i]
            c = att_outputs
        x = self.ln1(x + self.dropout(x_glo))
        x = self.ln2(x + self.dropout(self.fc(x)))
        return x

class STGformer2(nn.Module):
    def __init__(self, num_nodes, in_steps=12, out_steps=12, steps_per_day=288, input_dim=3, output_dim=1, input_embedding_dim=24,
        day_emb_dim=12, week_emb_dim=12, spatial_embedding_dim=0, adaptive_emb_dim=12, num_heads=4, supports=None, num_layers=3,
        dropout=0.1, mlp_ratio=2, dropout_a=0.3, kernel_size=[1], forward_transition=True, backward_transition=True):
        super(STGformer2, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_embedding_dim)
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.day_emb_dim = day_emb_dim
        self.week_emb_dim = week_emb_dim
        self.adaptive_emb_dim = adaptive_emb_dim
        self.steps_per_day = steps_per_day
        self.model_dim = input_embedding_dim + day_emb_dim + week_emb_dim + spatial_embedding_dim + adaptive_emb_dim
        self.forward_transition = forward_transition
        self.backward_transition = backward_transition
        if day_emb_dim > 0:
            self.day_emb = nn.Embedding(steps_per_day, day_emb_dim)
        if week_emb_dim > 0:
            self.week_emb = nn.Embedding(7, week_emb_dim)
        if adaptive_emb_dim > 0:
            self.adaptive_emb = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_emb_dim))
            )
        self.dropout = nn.Dropout(dropout_a)
        self.pooling = nn.AvgPool2d(kernel_size=(1, kernel_size[0]), stride=1)
        self.attn = nn.ModuleList([SelfAttention(self.model_dim, mlp_ratio, num_heads, dropout) for size in kernel_size])
        self.encoder1 = nn.Linear((in_steps - sum(k - 1 for k in kernel_size)) * self.model_dim, self.model_dim)
        self.encoder2 = nn.ModuleList([nn.Sequential(nn.Linear(self.model_dim, self.model_dim * mlp_ratio), nn.ReLU(), nn.Dropout(dropout), nn.Linear(self.model_dim * mlp_ratio, self.model_dim), nn.Dropout(dropout)) for _ in range(num_layers)])
        self.fc2 = nn.Linear(self.model_dim, out_steps * output_dim)
        self.conv = nn.Conv2d(self.model_dim, self.model_dim, (1, kernel_size[0]), 1, 0)

    def embedding_layer(self, x):
        batch_size = x.size(0)

        xx = x[..., :self.input_dim]
        xx = self.fc1(xx)
        features = torch.tensor([], device=x.device)
        
        if self.day_emb_dim > 0:
            day = x[..., 1]
            day_emb = self.day_emb((day * self.steps_per_day).long())
            features = torch.cat((features, day_emb), dim=-1)

        if self.day_emb_dim > 0:
            week = x[..., 2]
            week_emb = self.week_emb(week.long())
            features = torch.cat((features, week_emb), dim=-1)

        if self.adaptive_emb_dim > 0:
            adp_emb = self.adaptive_emb.expand(size=(batch_size, *self.adaptive_emb.shape))
            features = torch.cat((features, self.dropout(adp_emb)), dim=-1)   
        
        x = torch.cat([xx] + [features], dim=-1)
        x = self.conv(x.transpose(1,3)).transpose(1,3)

        return x

    def graph_propagation(self, x, graph, k, A):
        result = [x]
        A = torch.tensor(A, device=x.device)
        P_f = A/A.sum(dim=1)
        A_t = A.t()
        P_b = A_t/A_t.sum(dim=1)
        for i in range(1, k):
            x_f, x_b = 0, 0
            x_prev = result[-1]
            x_next = torch.matmul(graph, x_prev)
            if self.forward_transition:
                x_f = torch.matmul(P_f, x_prev)
            if self.backward_transition:
                x_b = torch.matmul(P_b, x_prev)
            result.append(x_next + x_f + x_b)
        return result


    def forward(self, x, A):
        batch_size = x.size(0)
        x = self.embedding_layer(x)
        graph = torch.matmul(self.adaptive_emb, self.adaptive_emb.transpose(1, 2)) #the self-adaptive adjacency matrix from "Graph WaveNet for Deep Spatial-Temporal Graph Modeling" (ends with softmax+relu)
        graph = self.pooling(graph.transpose(0, 2)).transpose(0, 2) #don't know why this is necessary
        graph = F.softmax(F.relu(graph), dim=-1)
        order = 2 #fixing order, but could be passed as argument
        for attn in self.attn:
            p = self.graph_propagation(x, graph, order, A)
            x = attn(x, p)
        x = self.encoder1(x.transpose(1, 2).flatten(-2)) #need to check why this is necessary (the transpose and flatten). actually need to check all this final part
        for layer in self.encoder2:
            x = x + layer(x)
        x = self.fc2(x).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
        x = x.transpose(1, 2)

        return x