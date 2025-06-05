import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class SAINTPlus(nn.Module):
    """
    SAINT+: Integrating Temporal Features for EdNet Correctness Prediction
    
    A Transformer-based knowledge tracing model with temporal features.
    """
    
    def __init__(self, 
                 num_questions: int = 9454,
                 num_concepts: int = 188,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_length: int = 200):
        
        super().__init__()
        
        self.d_model = d_model
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.max_seq_length = max_seq_length
        
        # Exercise embeddings for encoder
        self.question_embedding = nn.Embedding(num_questions + 1, d_model, padding_idx=0)
        self.concept_embedding = nn.Embedding(num_concepts + 1, d_model, padding_idx=0)
        self.part_embedding = nn.Embedding(8, d_model)
        
        # Response embeddings for decoder
        self.response_embedding = nn.Embedding(3, d_model)
        self.prior_response_embedding = nn.Embedding(3, d_model)
        
        # Temporal feature embeddings (SAINT+ innovation)
        self.elapsed_time_embedding = nn.Linear(1, d_model)
        self.lag_time_embedding = nn.Linear(1, d_model)
        
        # Additional features
        self.explanation_embedding = nn.Embedding(3, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
        """Create padding mask for sequences"""
        return (seq == pad_token)
    
    def create_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal mask to prevent looking at future tokens"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def encode_temporal_features(self, elapsed_time: torch.Tensor, lag_time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode temporal features using log scaling"""
        elapsed_time_log = torch.log(elapsed_time + 1e-10)
        lag_time_log = torch.log(lag_time + 1e-10)
        
        elapsed_embed = self.elapsed_time_embedding(elapsed_time_log.unsqueeze(-1))
        lag_embed = self.lag_time_embedding(lag_time_log.unsqueeze(-1))
        
        return elapsed_embed, lag_embed
    
    def forward(self, 
                questions: torch.Tensor,
                concepts: torch.Tensor,
                parts: torch.Tensor,
                responses: torch.Tensor,
                prior_responses: torch.Tensor,
                elapsed_times: torch.Tensor,
                lag_times: torch.Tensor,
                had_explanation: torch.Tensor) -> torch.Tensor:
        
        batch_size, seq_len = questions.shape
        device = questions.device
        
        # === ENCODER ===
        question_emb = self.question_embedding(questions)
        concept_emb = self.concept_embedding(concepts)
        part_emb = self.part_embedding(parts)
        explanation_emb = self.explanation_embedding(had_explanation)
        
        exercise_emb = question_emb + concept_emb + part_emb + explanation_emb
        exercise_emb = self.pos_encoding(exercise_emb.transpose(0, 1)).transpose(0, 1)
        exercise_emb = self.dropout(exercise_emb)
        
        encoder_padding_mask = self.create_padding_mask(questions)
        encoder_output = self.transformer_encoder(
            exercise_emb,
            src_key_padding_mask=encoder_padding_mask
        )
        
        # === DECODER ===
        response_emb = self.response_embedding(responses)
        prior_response_emb = self.prior_response_embedding(prior_responses)
        
        elapsed_emb, lag_emb = self.encode_temporal_features(elapsed_times, lag_times)
        
        decoder_input = response_emb + prior_response_emb + elapsed_emb + lag_emb
        decoder_input = self.pos_encoding(decoder_input.transpose(0, 1)).transpose(0, 1)
        decoder_input = self.dropout(decoder_input)
        
        causal_mask = self.create_causal_mask(seq_len).to(device)
        decoder_padding_mask = self.create_padding_mask(responses)
        
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_padding_mask,
            memory_key_padding_mask=encoder_padding_mask
        )
        
        decoder_output = self.layer_norm(decoder_output)
        logits = self.output_layer(decoder_output)
        predictions = torch.sigmoid(logits.squeeze(-1))
        
        return predictions 