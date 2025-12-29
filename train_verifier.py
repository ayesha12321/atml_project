import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

class ContrastivePairDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_length=512):
        self.pairs = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.pairs.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Tokenize question + solution_a
        text_a = f"Question: {pair['question']}\nSolution: {pair['solution_a']}"
        text_b = f"Question: {pair['question']}\nSolution: {pair['solution_b']}"
        
        encoding_a = self.tokenizer(
            text_a,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoding_b = self.tokenizer(
            text_b,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids_a': encoding_a['input_ids'].squeeze(0),
            'attention_mask_a': encoding_a['attention_mask'].squeeze(0),
            'input_ids_b': encoding_b['input_ids'].squeeze(0),
            'attention_mask_b': encoding_b['attention_mask'].squeeze(0),
            'label': torch.tensor(pair['label'], dtype=torch.float)
        }

class VerifierModel(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-base'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Simple MLP head
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        score = self.scorer(cls_embedding)
        return score.squeeze(-1)

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move to device
        input_ids_a = batch['input_ids_a'].to(device)
        attention_mask_a = batch['attention_mask_a'].to(device)
        input_ids_b = batch['input_ids_b'].to(device)
        attention_mask_b = batch['attention_mask_b'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        score_a = model(input_ids_a, attention_mask_a)
        score_b = model(input_ids_b, attention_mask_b)
        
        # Margin ranking loss: score_a should be > score_b by margin
        margin = 0.5
        loss = torch.clamp(margin - (score_a - score_b), min=0).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = (score_a > score_b).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids_a = batch['input_ids_a'].to(device)
            attention_mask_a = batch['attention_mask_a'].to(device)
            input_ids_b = batch['input_ids_b'].to(device)
            attention_mask_b = batch['attention_mask_b'].to(device)
            labels = batch['label'].to(device)
            
            score_a = model(input_ids_a, attention_mask_a)
            score_b = model(input_ids_b, attention_mask_b)
            
            margin = 0.5
            loss = torch.clamp(margin - (score_a - score_b), min=0).mean()
            
            total_loss += loss.item()
            predictions = (score_a > score_b).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

def main():
    # Config
    model_name = 'microsoft/deberta-v3-base'
    batch_size = 16
    epochs = 3
    learning_rate = 2e-5
    max_length = 512
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VerifierModel(model_name).to(device)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ContrastivePairDataset('train_pairs.jsonl', tokenizer, max_length)
    val_dataset = ContrastivePairDataset('val_pairs.jsonl', tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_name': model_name,
                'epoch': epoch,
                'val_acc': val_acc
            }, 'verifier_best.pt')
            print(f"Saved best model with val_acc: {val_acc:.4f}")
    
    print(f"\nTraining complete! Best val acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()