import torch.nn as nn
from peptide_dataset import *

class PeptideTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=21,
        d_model=128,
        nhead=4,
        num_layers=2,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()

        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=d_model * 4,
        #     dropout=dropout,
        #     batch_first=True,
        #     norm_first=False,
        # )
        #
        # self.decoder = nn.TransformerDecoder(
        #     decoder_layer,
        #     num_layers=num_layers,
        #     enable_nested_tensor=False,
        # )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, 2)

    def get_embeddings(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.size()

        if seq_len > self.max_len:
            raise ValueError(f"Длина {seq_len} превышает max_len={self.max_len}")

        positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).repeat(batch_size, 1)

        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        padding_mask = ~attention_mask.bool()

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        x = self.norm(x)

        mask = attention_mask.unsqueeze(-1).to(x.dtype)

        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)

        embeddings = summed / counts

        return embeddings

    def forward(self, input_ids, attention_mask):
        embeddings = self.get_embeddings(input_ids, attention_mask)

        logits = self.classifier(embeddings)

        return logits