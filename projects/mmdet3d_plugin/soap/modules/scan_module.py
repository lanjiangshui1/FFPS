import torch
from torch import nn


class ScanBlockLayer(nn.Module):
    def __init__(self, embed_dims, dim_feedforward, num_heads, target_axis="depth", dropout=0.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.target_axis = target_axis

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

        self.linear1 = nn.Linear(embed_dims, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dims)
        self.activation = nn.ReLU()
        self.drop2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, pos_embed=None):
        b, l, c = x.shape
        attn_mask = None
        if self.target_axis == "depth":
            attn_mask = torch.triu(torch.ones(l, l), diagonal=1) == 1
            attn_mask[:, : l // 2] = False
            attn_mask = attn_mask.to(x.device)
        elif self.target_axis == "height":
            attn_mask = torch.tril(torch.ones(l, l), diagonal=-1) == 1
            attn_mask = attn_mask.to(x.device)
        elif self.target_axis == "width":
            half_l = l // 2
            attn_quarter_mask = torch.tril(torch.ones(half_l, half_l), diagonal=-1) == 1
            attn_half_mask = torch.cat((attn_quarter_mask, attn_quarter_mask.flip(dims=[-1])), dim=-1)
            attn_mask = torch.cat((attn_half_mask, attn_half_mask.flip(dims=[-2])), dim=0)
            attn_mask[:, l // 4 : -(l // 4)] = False
            attn_mask = attn_mask.to(x.device)

        x_norm = self.norm1(x)
        x = x + self.drop1(
            self.attn(
                x_norm + pos_embed if pos_embed is not None else x_norm,
                x_norm + pos_embed if pos_embed is not None else x_norm,
                x_norm,
                attn_mask=attn_mask,
            )[0]
        )

        x1 = self.norm2(x)
        x = x + self.drop2(self.linear2(self.activation(self.linear1(x1))))
        return x


class ScanBlock(nn.Module):
    def __init__(self, embed_dims, num_heads=8, target_axis="depth"):
        super().__init__()
        self.embed_dims = embed_dims
        self.target_axis = target_axis
        self.scanblocklayer = ScanBlockLayer(
            embed_dims=embed_dims,
            dim_feedforward=embed_dims * 2,
            num_heads=num_heads,
            target_axis=target_axis,
            dropout=0.1,
        )

    def forward(self, scene_embed, pos_embed=None):
        # scene_embed: [H, W, Z, C]
        if pos_embed is not None:
            pos_embed = pos_embed.permute(0, 2, 3, 1)
        if self.target_axis == "depth":
            scene_embed = scene_embed.permute(1, 2, 0, 3)
            if pos_embed is not None:
                pos_embed = pos_embed.permute(1, 2, 0, 3)
        elif self.target_axis == "width":
            scene_embed = scene_embed.permute(0, 2, 1, 3)
            if pos_embed is not None:
                pos_embed = pos_embed.permute(0, 2, 1, 3)

        axis_1, axis_2, target_axis, dim = scene_embed.shape
        scene_embed = scene_embed.flatten(0, 1)
        if pos_embed is not None:
            pos_embed = pos_embed.flatten(0, 1)
        scene_embed = self.scanblocklayer(scene_embed, pos_embed)
        scene_embed = scene_embed.reshape(axis_1, axis_2, target_axis, dim)

        if self.target_axis == "depth":
            scene_embed = scene_embed.permute(2, 0, 1, 3)
        elif self.target_axis == "width":
            scene_embed = scene_embed.permute(0, 2, 1, 3)
        return scene_embed


class ScanModule(nn.Module):
    def __init__(self, embed_dims, num_heads=8):
        super().__init__()
        self.depth_block = ScanBlock(embed_dims, target_axis="depth", num_heads=num_heads)
        self.height_block = ScanBlock(embed_dims, target_axis="height", num_heads=num_heads)

    def forward(self, x, conf=None):
        # x: [B, C, X, Y, Z]
        if not torch.is_tensor(x):
            raise TypeError(f"ScanModule expects a Tensor, got {type(x)}")
        while x.dim() < 5:
            x = x.unsqueeze(0)
        if x.dim() == 5 and x.size(0) == 1:
            x = x.squeeze(0)
        if x.dim() != 4:
            raise RuntimeError(f"ScanModule expects 4D (C,X,Y,Z) or 5D (1,C,X,Y,Z), got shape {tuple(x.shape)}")
        c, xh, yw, zd = x.shape
        scene_embed = x.permute(1, 2, 3, 0).contiguous()
        depth_out = self.depth_block(scene_embed, pos_embed=None)
        height_out = self.height_block(scene_embed, pos_embed=None)
        out = (depth_out + height_out) * 0.5
        out = out.permute(3, 0, 1, 2).unsqueeze(0).contiguous()

        if conf is not None:
            out = conf * out + (1.0 - conf) * x
        return out
