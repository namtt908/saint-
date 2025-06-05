import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class KnowledgeTracingDataset(Dataset):
    """
    Dataset class for knowledge tracing data.
    
    This dataset handles the preprocessing and batching of student interaction sequences
    for the SAINT+ model.
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 sequence_length: int = 200,
                 pad_token: int = 0):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame containing student interaction data
            sequence_length: Maximum sequence length for padding/truncating
            pad_token: Token used for padding sequences
        """
        self.data = data
        self.sequence_length = sequence_length
        self.pad_token = pad_token
        
        # Group data by user_id to create sequences
        self.sequences = self._create_sequences()
        
    def _create_sequences(self) -> List[Dict[str, torch.Tensor]]:
        """Create sequences from the raw data."""
        sequences = []
        
        for user_id, group in self.data.groupby('user_id'):
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Create sequence
            sequence = {
                'questions': torch.tensor(group['question_id'].values, dtype=torch.long),
                'concepts': torch.tensor(group['concept_id'].values, dtype=torch.long),
                'parts': torch.tensor(group['part'].values, dtype=torch.long),
                'responses': torch.tensor(group['correct'].values, dtype=torch.long),
                'elapsed_times': torch.tensor(group['elapsed_time'].values, dtype=torch.float),
                'lag_times': torch.tensor(group['lag_time'].values, dtype=torch.float),
                'had_explanation': torch.tensor(group['had_explanation'].values, dtype=torch.long)
            }
            
            # Add prior responses
            sequence['prior_responses'] = torch.cat([
                torch.tensor([self.pad_token], dtype=torch.long),
                sequence['responses'][:-1]
            ])
            
            sequences.append(sequence)
            
        return sequences
    
    def _pad_sequence(self, sequence: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pad or truncate sequence to fixed length."""
        seq_len = len(sequence['questions'])
        
        if seq_len > self.sequence_length:
            # Truncate
            for key in sequence:
                sequence[key] = sequence[key][:self.sequence_length]
        elif seq_len < self.sequence_length:
            # Pad
            pad_len = self.sequence_length - seq_len
            for key in sequence:
                if key in ['elapsed_times', 'lag_times']:
                    # Pad with zeros for time features
                    pad = torch.zeros(pad_len, dtype=sequence[key].dtype)
                else:
                    # Pad with pad_token for categorical features
                    pad = torch.full((pad_len,), self.pad_token, dtype=sequence[key].dtype)
                sequence[key] = torch.cat([sequence[key], pad])
        
        return sequence
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence by index."""
        sequence = self.sequences[idx]
        return self._pad_sequence(sequence)
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        return {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0].keys()
        } 