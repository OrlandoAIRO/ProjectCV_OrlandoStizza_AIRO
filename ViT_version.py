import torch
import torch.nn as nn

# =====================================================
# Patch Embedding
# =====================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                 in_channels=3, embed_dim=512):
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# =====================================================
# Transformer Encoder Block
# =====================================================

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512,
                 num_heads=8, mlp_ratio=4,
                 dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        attn_out, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x)
        )

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))

        return x


# =====================================================
# Multi-Task Vision Transformer
# =====================================================

class MultiTaskViT(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 embed_dim=512,
                 depth=6,
                 num_heads=8,
                 dropout=0.1,
                 num_real_fake=2,
                 num_transform=3):

        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size,
            patch_size,
            3,
            embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # =================================================
        # Multi-task heads
        # =================================================

        self.real_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_real_fake)
        )

        self.transform_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_transform)
        )

    def forward(self, x):

        B = x.shape[0]

        # Patch tokens
        x = self.patch_embed(x)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer encoder
        for block in self.encoder:
            x = block(x)

        x = self.norm(x)

        # CLS feature
        features = x[:, 0]

        # Task heads
        real_logits = self.real_head(features)
        transform_logits = self.transform_head(features)

        return real_logits, transform_logits


# =====================================================
# Initialization
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MultiTaskViT().to(device)

criterion_real = nn.CrossEntropyLoss()
criterion_transform = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

lambda_real = 1.0
lambda_transform = 1.0


# =====================================================
# Training Step
# =====================================================

def training_step(images,
                  real_labels,
                  transform_labels):

    images = images.to(device)
    real_labels = real_labels.to(device)
    transform_labels = transform_labels.to(device)

    optimizer.zero_grad()

    real_logits, transform_logits = model(images)

    loss_real = criterion_real(
        real_logits,
        real_labels
    )

    loss_transform = criterion_transform(
        transform_logits,
        transform_labels
    )

    total_loss = (
        lambda_real * loss_real +
        lambda_transform * loss_transform
    )

    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "real_loss": loss_real.item(),
        "transform_loss": loss_transform.item()
    }