# settings.py
# Cesty
TRAINING_TEXT_DIR = "training_text"  # Adresář s trénovacími texty
MODEL_PREFIX = "sp_model"  # Předpona pro SentencePiece model
WEIGHTS_DIR = "saved_weights"  # Adresář pro uložení vah modelu

# Parametry datasetu a tokenizace
SEQ_LENGTH = 10  # Délka vstupní sekvence
VOCAB_SIZE = 10000  # Velikost slovníku (sníženo kvůli předchozí chybě)

# Parametry modelu
EMBEDDING_DIM = 200  # Dimenze embeddingů
HIDDEN_DIM = 256  # Dimenze skrytých stavů
NUM_LAYERS = 2  # Počet vrstev LSTM
DROPOUT = 0.3  # Dropout pro regularizaci

# Parametry trénování
EPOCHS = 10  # Počet epoch
BATCH_SIZE = 32  # Velikost dávky
LEARNING_RATE = 0.002  # Rychlost učení
WEIGHT_DECAY = 1e-4  # L2 regularizace
PATIENCE = 3  # Počet epoch bez zlepšení pro early stopping
SCHEDULER_PATIENCE = 2  # Počet epoch bez zlepšení pro scheduler
SCHEDULER_FACTOR = 0.5  # Faktor zmenšení LR v scheduleru
TRAIN_SPLIT = 0.8  # Poměr trénovací části datasetu

# Parametry beam search
BEAM_WIDTH = 5  # Šířka beam search
NUM_WORDS = 10  # Počet slov k predikci

# SentencePiece parametry
CHARACTER_COVERAGE = 0.9995  # Pokrytí znaků pro SentencePiece
MODEL_TYPE = "bpe"  # Typ modelu (BPE, unigram apod.)
