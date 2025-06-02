import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=33):  
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CaptionGenerator(nn.Module):  # <-- renamed from generate_caption
    def __init__(self, embed_size, vocab_size, max_seq_len, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout=dropout, max_len=max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.last_linear_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, img_features, tgt_seq):
        tgt_emb = self.embedding(tgt_seq)
        tgt_pos = self.pos_encoder(tgt_emb)
        memory = img_features.unsqueeze(1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq.size(1)).to(tgt_seq.device)
        output = self.TransformerDecoder(tgt_pos, memory, tgt_mask=tgt_mask)
        return self.last_linear_layer(output)
