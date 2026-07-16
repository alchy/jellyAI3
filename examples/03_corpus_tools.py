"""03 — Korpusové nástroje se start/stop (potřebuje ÚFAL modely).

Ukazuje explicitní životní cyklus: `with jellyai.CorpusTools() as tools:` služby
nastartuje a na konci složí. Modely stáhni: ./jelly qa-models
Spusť: python examples/03_corpus_tools.py
"""

import jellyai

try:
    with jellyai.CorpusTools() as tools:
        parsed = tools.parse("Karel Čapek se narodil roku 1890.")
        for sentence in parsed:
            for token in sentence:
                print(f"{token['form']:<12} {token['lemma']:<12} {token['deprel']}")
except jellyai.JellyError as err:
    print("Chyba:", err)
except Exception as err:  # pylint: disable=broad-exception-caught
    print("Nejspíš chybí modely (./jelly qa-models):", err)
