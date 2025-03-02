import torch
import torch.nn as nn
from settings import EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT

class LSTMModel(nn.Module):
    """
    LSTM model pro predikci dalšího slova v sekvenci.
    """
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        """
        Inicializace modelu.

        Args:
            vocab_size (int): Velikost slovníku.
            embedding_dim (int): Dimenze embeddingů.
            hidden_dim (int): Dimenze skrytých stavů.
            num_layers (int): Počet vrstev LSTM.
            dropout (float): Hodnota dropout pro regularizaci.
        """
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # Dropout pro regularizaci
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        Předávací metoda modelu.

        Args:
            x (torch.Tensor): Vstupní sekvence.

        Returns:
            torch.Tensor: Predikce dalšího slova.
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        output = self.linear(last_out)
        return output