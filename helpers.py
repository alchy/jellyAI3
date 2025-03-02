import torch
import unicodedata
import os
import re
from settings import SEQ_LENGTH, BEAM_WIDTH, NUM_WORDS, TRAINING_TEXT_DIR

def remove_diacritics(text):
    """
    Odstraní diakritiku z českého textu.
    
    Args:
        text (str): Vstupní text.
    
    Returns:
        str: Text bez diakritiky.
    """
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    return text

def load_texts_from_directory(directory=TRAINING_TEXT_DIR):
    """
    Načte textové soubory (.txt) z adresáře, odstraní diakritiku, převede na malá písmena,
    odstraní interpunkci a normalizuje čísla.
    
    Args:
        directory (str): Cesta k adresáři.
    
    Returns:
        str: Normalizovaný text ze všech souborů.
    """
    all_text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                text = remove_diacritics(text)
                text = text.lower()
                text = re.sub(r'[^\w\s]', '', text)  # Odstranění interpunkce
                text = re.sub(r'\d+', '<NUM>', text)  # Nahrazení čísel tokenem <NUM>
                all_text += text + " "
    return all_text.strip()

def beam_search(model, dataset, input_text, num_words=NUM_WORDS, seq_length=SEQ_LENGTH, beam_width=BEAM_WIDTH, device=None):
    """
    Použije beam search pro predikci dalších slov.
    
    Args:
        model: Trénovaný LSTM model.
        dataset: TextDataset instance.
        input_text (str): Vstupní text od uživatele.
        num_words (int): Počet slov k predikci.
        seq_length (int): Délka vstupní sekvence.
        beam_width (int): Šířka beam search.
        device: Zařízení (CPU nebo GPU).
    
    Returns:
        list: Seznam predikovaných tokenů.
    """
    model.eval()
    # Tokenizace vstupního textu na subword jednotky
    tokens = dataset.sp.encode_as_ids(input_text)
    # Validate tokens to prevent out-of-range indices
    tokens = [t if t < dataset.vocab_size else dataset.sp.unk_id() for t in tokens]  # Replace OOV with <unk>
    if len(tokens) > seq_length:
        tokens = tokens[-seq_length:]
    elif len(tokens) < seq_length:
        tokens = [dataset.sp.pad_id()] * (seq_length - len(tokens)) + tokens  # Použití padding tokenu
    
    beams = [(tokens, 0.0)]  # Inicializace s počáteční sekvencí a skóre 0
    
    for step in range(num_words):
        new_beams = []
        for beam in beams:
            current_sequence, current_scor
