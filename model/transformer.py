
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:x.size(1)].unsqueeze(0)
    

class AgentTransformer(nn.Module):
    def __init__(self, input_dim=6, model_dim=128, num_heads=4, num_layers=2, num_agents=50, future_steps=60):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.agent_embedding = nn.Embedding(num_agents, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.head = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.ReLU(),
            nn.Linear(128, future_steps * 2)  # Predict 60 steps of (x, y)
        )

    def forward(self, data):
        """
        x: shape (B, A, T, F) → input past for all agents
        """
        x = data.x
        x = x.view(-1, 50, 50, 6)  # (B, A, T, F) where F=6
        B, A, T, F = x.shape
        
        # Add agent embedding
        agent_ids = torch.arange(A, device=x.device).view(1, A, 1).expand(B, A, 1)
        agent_embed = self.agent_embedding(agent_ids).expand(-1, -1, T, -1)  # (B, A, T, D)

       
        x = x.view(B*A*T, -1) # Flatten to (B*A*T, F)
        x = self.input_proj(x)  # (B, A, T, D)
        x = x.view(B, A, T, -1)
        x = x + agent_embed

        x = x.view(B * A, T, -1)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (B*A, T, D)
        x = x.view(B, A, T, -1)

        ego_feat = x[:, 0]  # only agent 0
        pooled = ego_feat.mean(dim=1)  # (B, D)

        out = self.head(pooled).view(B, -1, 2)  # (B, 60, 2)
        return out
