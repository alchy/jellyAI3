import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from settings import SEQ_LENGTH, MODEL_PREFIX

class TextDataset(Dataset):
    def __init__(self, text):
        """
        Inicializace datasetu s subword tokenizací.

        Args:
            text (str): Vstupní text.
        """
        self.seq_length = SEQ_LENGTH
        # Načtení SentencePiece modelu
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{MODEL_PREFIX}.model")
        print("Loaded SentencePiece vocab size:", self.sp.get_piece_size())  # Debug print
        
        # Tokenizace textu na subword jednotky
        self.tokens = self.sp.encode_as_ids(text)
        self.vocab_size = self.sp.get_piece_size()  # Velikost slovníku z modelu

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        inputs = torch.tensor(self.tokens[idx:idx + self.seq_length], dtype=torch.long)
        target = torch.tensor(self.tokens[idx + self.seq_length], dtype=torch.long)
        return inputs, target

    def tokens_to_text(self, tokens):
        """
        Převod tokenů zpět na text.
        """
        return self.sp.decode_ids(tokens)
    