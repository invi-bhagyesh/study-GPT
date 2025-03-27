import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import GPT2Config
from model.gpt2 import GPT2
import wandb
from tqdm import tqdm

def train_gpt2(
    train_loader,
    val_loader=None,
    n_epochs=3,
    learning_rate=3e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # Initialize config and model
    config = GPT2Config()
    model = GPT2(config).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize wandb
    wandb.init(project="gpt2-pretraining", config={
        "learning_rate": learning_rate,
        "epochs": n_epochs,
        "batch_size": train_loader.batch_size,
        "model_config": config.__dict__
    })
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate loss
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            wandb.log({"batch_loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        wandb.log({"epoch": epoch, "train_loss": avg_loss})
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    
                    logits = model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            wandb.log({"val_loss": avg_val_loss})
            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

    wandb.finish()
    return model