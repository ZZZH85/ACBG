import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Linear Embedding:
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class Expert(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(Expert, self).__init__()
        self.mlp = MLP(input_dim, embed_dim)

    def forward(self, x):
        return self.mlp(x)


class GatingMechanism(nn.Module):
    def __init__(self, embed_dim):
        super(GatingMechanism, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, experts_outputs):
        # Concatenate the outputs of all experts
        combined = torch.cat(experts_outputs, dim=1)
        # Apply the gating mechanism
        gates = self.gate(combined)
        return gates


class DecoderHead(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], num_classes=40,
                 dropout_ratio=0.1, norm_layer=nn.BatchNorm2d, embed_dim=768,
                 align_corners=False):
        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.experts = nn.ModuleList([
            Expert(c4_in_channels, embedding_dim),
            Expert(c3_in_channels, embedding_dim),
            Expert(c2_in_channels, embedding_dim),
            Expert(c1_in_channels, embedding_dim)
        ])

        self.gating_mechanism = GatingMechanism(embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, return_feats=False):
        c1, c2, c3, c4 = inputs

        experts_outputs = []
        for i, expert in enumerate(self.experts):
            if i == 0:
                x = expert(c4).permute(0, 2, 1).reshape(c4.size(0), -1, c4.size(2), c4.size(3))
                x = F.interpolate(x, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
            elif i == 1:
                x = expert(c3).permute(0, 2, 1).reshape(c3.size(0), -1, c3.size(2), c3.size(3))
                x = F.interpolate(x, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
            elif i == 2:
                x = expert(c2).permute(0, 2, 1).reshape(c2.size(0), -1, c2.size(2), c2.size(3))
                x = F.interpolate(x, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
            else:
                x = expert(c1).permute(0, 2, 1).reshape(c1.size(0), -1, c1.size(2), c1.size(3))
            experts_outputs.append(x)

        gates = self.gating_mechanism(experts_outputs)
        gated_outputs = [g * x for g, x in zip(gates, experts_outputs)]
        combined = sum(gated_outputs)

        x = self.linear_fuse(combined)
        x = self.dropout(x)
        x = self.linear_pred(x)

        if return_feats:
            return x, combined
        else:
            return x