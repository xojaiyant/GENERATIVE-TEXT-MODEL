# Text Generation using GPT-2
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate text
def generate_text(prompt, max_length=150, temperature=0.7):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            num_return_sequences=1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# -----------------------------------
# Try your own prompt here:
prompt = input("Enter a topic or prompt: ")

generated = generate_text(prompt)
print("\nGenerated Text:\n")
print(generated)
