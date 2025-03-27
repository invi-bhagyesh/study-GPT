from transformers import GPT2Tokenizer
from data.dataset import TextDataset
from train import train_gpt2

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare your text data
texts = [...] # Your text data here

# Create dataset
dataset = TextDataset(texts, tokenizer)

# Train the model
model = train_gpt2(
    train_dataset=dataset,
    n_epochs=3,
    batch_size=8,
    learning_rate=3e-4
)

# Save the model
torch.save(model.state_dict(), 'gpt2_model.pt') 