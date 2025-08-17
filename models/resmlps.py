import torch
import torch.nn as nn

from models import register


@register('resmlps')
class RESMLPS(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list,coord_hidden_list):
        super().__init__()

        self.mlps = [RES_MLP(in_dim, out_dim, coord_hidden_list) for _ in range(10)]

        self.cood_mlp = COOD_MLP(2, 16, hidden_list)

        self.expansionNet = ExpansionNet(580, 10, hidden_list)

    def forward(self, x, rel_coord):
        scores = self.expansionNet(x)

        cood_out = self.cood_mlp(rel_coord)

        res_list = [mlp(x, cood_out).cpu().detach().numpy() for mlp in self.mlps]

        mlp_res = torch.tensor(res_list).cuda()

        mlp_res = mlp_res.transpose(0, 1)

        return torch.sum(scores.unsqueeze(-1) * mlp_res, 1)


class RES_MLP(nn.Module):

    def __init__(self, in_dim, out_dim, coord_hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in coord_hidden_list:
            layers.append(nn.Linear(lastv, 16).cuda())
            lastv = hidden
        self.layers = nn.Sequential(*layers)
        self.relu = nn.ReLU()

        self.linear_out = nn.Linear(lastv, out_dim).cuda()

    def forward(self, x, cood_out):
        out = x.view(-1, x.shape[-1])

        for layer in self.layers:
            out = self.relu(layer(out))
            out = torch.add(out, cood_out)
        out = self.linear_out(out)
        shape = out.shape[:-1]
        return out.view(*shape, -1)


class COOD_MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class ExpansionNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super(ExpansionNet, self).__init__()
        layers = []
        lastv = in_dim
        self.out_dim = out_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        b, c = x.shape
        x = x.view(-1, c)
        logits = self.layers(x)
        out = nn.functional.normalize(logits, dim=1)
        return out.view(b, self.out_dim)
