"""SentencePiece BPE tokenizer pro generátor.

Než může transformer cokoli generovat, potřebuje text rozsekat na kousky (tokeny)
a naopak z tokenů složit text. Používáme SentencePiece BPE — naučí se podslovní
jednotky přímo z korpusu, takže zvládne i tvary, co ve slovníku nejsou (díky
`byte_fallback` neselže na žádném znaku). `<eos>` (konec sekvence) je vestavěný
řídicí token SentencePiece; podle něj model pozná, kde odpověď skončit.
"""

import sentencepiece as spm


def train_tokenizer(corpus_path, prefix, vocab_size, character_coverage=1.0):
    """Natrénuje SentencePiece BPE model nad korpusem a uloží ho.

    Args:
        corpus_path (str): Cesta k textovému korpusu (jeden nebo víc řádků).
        prefix (str): Předpona výstupu — vzniknou `prefix.model` a `prefix.vocab`.
        vocab_size (int): Cílová velikost slovníku.
        character_coverage (float): Pokrytí znaků (1.0 pro češtinu = všechny znaky).

    Returns:
        str: Cesta k natrénovanému `.model` souboru.
    """
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        pad_id=3,                    # zapneme <pad> (výchozí je vypnutý)
        byte_fallback=True,          # ať nikdy neselže na neznámém znaku
        hard_vocab_limit=False,      # menší korpus nesmí shodit trénink na vocab_size
        max_sentence_length=100000,  # nepřeskakovat dlouhé řádky/odstavce (default 4192 B)
    )
    return prefix + ".model"


class SPTokenizer:
    """Obal nad SentencePiece procesorem s pohodlným rozhraním.

    Atributy:
        eos_id (int): Id tokenu konce sekvence.
        pad_id (int): Id výplňového tokenu.
        vocab_size (int): Skutečná velikost slovníku.
    """

    def __init__(self, sp):
        self._sp = sp
        self.eos_id = sp.eos_id()
        self.pad_id = sp.pad_id()
        self.vocab_size = sp.get_piece_size()

    @classmethod
    def load(cls, prefix):
        """Načte natrénovaný tokenizer.

        Args:
            prefix (str): Předpona (`prefix`) nebo přímo cesta k `.model`.

        Returns:
            SPTokenizer: Připravený tokenizer.
        """
        path = prefix if prefix.endswith(".model") else prefix + ".model"
        sp = spm.SentencePieceProcessor()
        sp.load(path)
        return cls(sp)

    def encode(self, text):
        """Převede text na seznam id tokenů.

        Args:
            text (str): Vstupní text.

        Returns:
            list[int]: Id tokenů.
        """
        return self._sp.encode(text, out_type=int)

    def decode(self, ids):
        """Složí z id tokenů zpět text.

        Args:
            ids (list[int]): Id tokenů.

        Returns:
            str: Rekonstruovaný text.
        """
        return self._sp.decode(list(ids))
