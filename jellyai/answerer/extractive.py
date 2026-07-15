"""Extraktivní answerer — odpověď je věta vytažená přímo z textu.

Nejpoctivější způsob, jak odpovědět „podle knihy": nevymýšlet, ale ukázat prstem
na konkrétní větu, která se dotazu nejvíc týká. Nevznikne tím elegantní souvětí
(od toho bude ve V2 generátor), zato odpověď nikdy nezhalucinuje — pochází
doslova z pramene a nese s sebou zdroj. Klasika žánru „radši nudná pravda než
poutavá lež".
"""

from jellyai.answerer.base import Answer, Answerer
from jellyai.text import tokenize, split_sentences, CZECH_STOPWORDS

# Hláška, když se nic relevantního nenajde. Přiznat nevědomost je feature, ne bug.
_NO_ANSWER = "V textu jsem odpověď nenašel."


class ExtractiveAnswerer(Answerer):
    """Vybere z nalezených pasáží jednu nejrelevantnější větu.

    Skóre věty = kolik slov sdílí s dotazem, podělené odmocninou délky věty
    (aby dlouhé věty nevyhrávaly jen proto, že mají víc slov a tím víc náhodných
    shod). Vrací se věta s nejvyšším skóre a její zdroj.
    """

    def __init__(self, config):
        """Vytvoří answerer s danou konfigurací.

        Args:
            config (AnswererConfig): Nastavení — hlavně `template` (zda odpověď
                zabalit do věty „Podle textu: …").
        """
        self.config = config

    def answer(self, question, retrieved):
        """Najde a vrátí větu nejlépe odpovídající dotazu.

        Projde věty všech nalezených pasáží, každou ohodnotí podle překryvu slov
        s dotazem (bez stopslov) a vybere nejlepší. Když žádná věta nesdílí s
        dotazem ani jedno slovo (nebo retriever nic nevrátil), poctivě přizná,
        že odpověď nenašel.

        Args:
            question (str): Dotaz uživatele.
            retrieved (list[tuple[Passage, float]]): Pasáže a skóre z retrieveru.

        Returns:
            Answer: Nejlepší věta jako `text`, zdroj `doc_id#index` v `sources`
                a skóre shody; při neúspěchu prázdné zdroje a hláška o nenalezení.
        """
        if not retrieved:
            return Answer(text=_NO_ANSWER, sources=[], score=0.0)

        q_tokens = set(tokenize(question, CZECH_STOPWORDS))
        best = None  # (skóre, věta, zdroj)
        for passage, _pscore in retrieved:
            for sent in split_sentences(passage.text):
                s_tokens = tokenize(sent, CZECH_STOPWORDS)
                if not s_tokens:
                    continue
                overlap = sum(1 for t in s_tokens if t in q_tokens)
                if overlap == 0:
                    continue
                score = overlap / (len(s_tokens) ** 0.5)
                source = f"{passage.doc_id}#{passage.index}"
                if best is None or score > best[0]:
                    best = (score, sent, source)

        if best is None:
            return Answer(text=_NO_ANSWER, sources=[], score=0.0)

        score, sentence, source = best
        text = f"Podle textu: {sentence}" if self.config.template else sentence
        return Answer(text=text, sources=[source], score=float(score))


def explain():
    """Vrátí lidský popis bloku ExtractiveAnswerer pro výukovou vrstvu.

    Returns:
        str: Několikavětý popis toho, jak se odpověď vybírá a proč nese zdroj.
    """
    return (
        "ExtractiveAnswerer vezme pasáže od Retrieveru, rozdělí je na věty a vybere "
        "tu s největším překryvem slov s dotazem (normalizováno délkou věty). "
        "Vrací větu doslova z textu + zdroj (doc_id#index), takže odpověď je "
        "dohledatelná a nevymýšlí si. Volitelná šablona přidá prefix 'Podle textu:'."
    )
