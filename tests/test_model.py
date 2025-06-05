import torch
import pytest
from saint_plus import SAINTPlus

def test_model_initialization():
    """Test model initialization with default parameters."""
    model = SAINTPlus()
    assert isinstance(model, SAINTPlus)
    assert model.d_model == 256
    assert model.num_questions == 9454
    assert model.num_concepts == 188

def test_model_forward():
    """Test model forward pass."""
    batch_size = 2
    seq_length = 10
    model = SAINTPlus(d_model=64)  # Use smaller model for testing
    
    # Create dummy input
    inputs = {
        'questions': torch.randint(0, 9454, (batch_size, seq_length)),
        'concepts': torch.randint(0, 188, (batch_size, seq_length)),
        'parts': torch.randint(0, 8, (batch_size, seq_length)),
        'responses': torch.randint(0, 2, (batch_size, seq_length)),
        'prior_responses': torch.randint(0, 2, (batch_size, seq_length)),
        'elapsed_times': torch.rand(batch_size, seq_length),
        'lag_times': torch.rand(batch_size, seq_length),
        'had_explanation': torch.randint(0, 2, (batch_size, seq_length))
    }
    
    # Forward pass
    outputs = model(**inputs)
    
    # Check output shape and values
    assert outputs.shape == (batch_size, seq_length)
    assert torch.all((outputs >= 0) & (outputs <= 1))  # Check probabilities

def test_model_padding():
    """Test model handling of padding."""
    batch_size = 2
    seq_length = 10
    model = SAINTPlus(d_model=64)
    
    # Create input with padding
    inputs = {
        'questions': torch.randint(0, 9454, (batch_size, seq_length)),
        'concepts': torch.randint(0, 188, (batch_size, seq_length)),
        'parts': torch.randint(0, 8, (batch_size, seq_length)),
        'responses': torch.randint(0, 2, (batch_size, seq_length)),
        'prior_responses': torch.randint(0, 2, (batch_size, seq_length)),
        'elapsed_times': torch.rand(batch_size, seq_length),
        'lag_times': torch.rand(batch_size, seq_length),
        'had_explanation': torch.randint(0, 2, (batch_size, seq_length))
    }
    
    # Add padding
    inputs['questions'][:, -2:] = 0
    inputs['concepts'][:, -2:] = 0
    inputs['parts'][:, -2:] = 0
    inputs['responses'][:, -2:] = 0
    inputs['prior_responses'][:, -2:] = 0
    inputs['elapsed_times'][:, -2:] = 0
    inputs['lag_times'][:, -2:] = 0
    inputs['had_explanation'][:, -2:] = 0
    
    # Forward pass
    outputs = model(**inputs)
    
    # Check output shape
    assert outputs.shape == (batch_size, seq_length)

def test_model_device():
    """Test model device handling."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = SAINTPlus(d_model=64).to(device)
        
        # Create dummy input
        batch_size = 2
        seq_length = 10
        inputs = {
            'questions': torch.randint(0, 9454, (batch_size, seq_length), device=device),
            'concepts': torch.randint(0, 188, (batch_size, seq_length), device=device),
            'parts': torch.randint(0, 8, (batch_size, seq_length), device=device),
            'responses': torch.randint(0, 2, (batch_size, seq_length), device=device),
            'prior_responses': torch.randint(0, 2, (batch_size, seq_length), device=device),
            'elapsed_times': torch.rand(batch_size, seq_length, device=device),
            'lag_times': torch.rand(batch_size, seq_length, device=device),
            'had_explanation': torch.randint(0, 2, (batch_size, seq_length), device=device)
        }
        
        # Forward pass
        outputs = model(**inputs)
        
        # Check output device
        assert outputs.device == device 