import sentencepiece as spm
from helpers import load_texts_from_directory
import os
from settings import TRAINING_TEXT_DIR, VOCAB_SIZE, MODEL_PREFIX, CHARACTER_COVERAGE, MODEL_TYPE

def train_sentencepiece_model():
    """
    Trénuje SentencePiece model pro subword tokenizaci.
    """
    # Načtení textu z adresáře
    text = load_texts_from_directory(TRAINING_TEXT_DIR)
    if not text.strip():
        raise ValueError("Žádný text nebyl načten z adresáře 'training_text'. Zkontrolujte, zda obsahuje .txt soubory s daty.")
    
    with open("temp_training_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    # Trénování SentencePiece modelu
    spm.SentencePieceTrainer.train(
        input="temp_training_text.txt",
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        character_coverage=CHARACTER_COVERAGE,
        model_type=MODEL_TYPE
    )
    print(f"SentencePiece model uložen jako {MODEL_PREFIX}.model a {MODEL_PREFIX}.vocab")
    os.remove("temp_training_text.txt")  # Odstranění dočasného souboru

if __name__ == "__main__":
    train_sentencepiece_model()
    