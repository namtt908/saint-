import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from .data.dataset import KnowledgeTracingDataset
from .utils.metrics import compute_metrics

class SAINTPlusTrainer:
    """
    Trainer class for SAINT+ model.
    
    Handles model training, validation, and evaluation.
    """
    
    def __init__(self,
                 model: nn.Module,
                 config_path: str,
                 device: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            model: SAINT+ model instance
            config_path: Path to configuration file
            device: Device to use for training (cuda/cpu)
        """
        self.model = model
        self.config = self._load_config(config_path)
        self.device = device or self.config['training']['device']
        self.model = self.model.to(self.device)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        # Initialize loss function
        self.criterion = nn.BCELoss()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config['training']['output_dir']) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc='Training'):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(
                questions=batch['questions'],
                concepts=batch['concepts'],
                parts=batch['parts'],
                responses=batch['responses'],
                prior_responses=batch['prior_responses'],
                elapsed_times=batch['elapsed_times'],
                lag_times=batch['lag_times'],
                had_explanation=batch['had_explanation']
            )
            
            # Compute loss
            loss = self.criterion(predictions, batch['responses'].float())
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            all_preds.append(predictions.detach())
            all_labels.append(batch['responses'])
        
        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(
            all_labels,
            all_preds,
            threshold=self.config['evaluation']['threshold']
        )
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.
        
        Args:
            eval_loader: DataLoader for evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(eval_loader, desc='Evaluating'):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(
                questions=batch['questions'],
                concepts=batch['concepts'],
                parts=batch['parts'],
                responses=batch['responses'],
                prior_responses=batch['prior_responses'],
                elapsed_times=batch['elapsed_times'],
                lag_times=batch['lag_times'],
                had_explanation=batch['had_explanation']
            )
            
            # Compute loss
            loss = self.criterion(predictions, batch['responses'].float())
            
            # Update metrics
            total_loss += loss.item()
            all_preds.append(predictions)
            all_labels.append(batch['responses'])
        
        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(
            all_labels,
            all_preds,
            threshold=self.config['evaluation']['threshold']
        )
        metrics['loss'] = total_loss / len(eval_loader)
        
        return metrics
    
    def train(self,
              train_dataset: KnowledgeTracingDataset,
              val_dataset: Optional[KnowledgeTracingDataset] = None,
              test_dataset: Optional[KnowledgeTracingDataset] = None):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            test_dataset: Test dataset (optional)
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            collate_fn=KnowledgeTracingDataset.collate_fn
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=self.config['training']['num_workers'],
                collate_fn=KnowledgeTracingDataset.collate_fn
            )
        
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=self.config['training']['num_workers'],
                collate_fn=KnowledgeTracingDataset.collate_fn
            )
        
        # Training loop
        best_val_auc = 0
        for epoch in range(self.config['training']['num_epochs']):
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.logger.info(f"Train metrics: {train_metrics}")
            
            # Validate
            if val_dataset:
                val_metrics = self.evaluate(val_loader)
                self.logger.info(f"Validation metrics: {val_metrics}")
                
                # Save best model
                if val_metrics['auc'] > best_val_auc:
                    best_val_auc = val_metrics['auc']
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_metrics)
        
        # Test
        if test_dataset:
            test_metrics = self.evaluate(test_loader)
            self.logger.info(f"Test metrics: {test_metrics}")
    
    def save_checkpoint(self,
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = Path(self.config['training']['output_dir']) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_model_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics'] 