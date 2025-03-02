import torch
from dataset import TextDataset
from model import LSTMModel
from train import train_model
from helpers import load_texts_from_directory, beam_search, remove_diacritics
from torch.utils.data import DataLoader, random_split
from settings import *
import re

# Zařízení
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Načtení textů z adresáře
input_text = load_texts_from_directory(TRAINING_TEXT_DIR)

# Vytvoření datasetu
dataset = TextDataset(input_text)

# Rozdělení datasetu na trénovací a validační část
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Vytvoření DataLoaderů
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Inicializace modelu
model = LSTMModel(
    vocab_size=dataset.vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)
print("Model vocab size:", model.embedding.num_embeddings)  # Debug print

# Trénování modelu
final_loss = train_model(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE)

# Interakce s uživatelem
while True:
    user_input = input("Zadejte text pro predikci (nebo 'exit' pro ukončení): ")
    if user_input.lower() == 'exit':
        break
    # Preprocess input
    user_input = remove_diacritics(user_input).lower()
    user_input = re.sub(r'[^\w\s]', '', user_input)  # Remove punctuation
    user_input = re.sub(r'\d+', '<NUM>', user_input)  # Replace numbers
    # Debug tokenization
    tokens = dataset.sp.encode_as_ids(user_input)
    print("Input text:", user_input)
    print("Tokenized IDs:", tokens)
    print("Max token ID:", max(tokens) if tokens else "Empty")
    print("Vocab size:", dataset.vocab_size)
    predicted_tokens = beam_search(
        model, dataset, user_input,
        num_words=NUM_WORDS, seq_length=SEQ_LENGTH,
        beam_width=BEAM_WIDTH, device=DEVICE
    )
    predicted_text = dataset.tokens_to_text(predicted_tokens)
    print("Predikovaný text:", predicted_text)
    