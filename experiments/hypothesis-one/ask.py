#!/usr/bin/env python3
"""Interaktivní dotaz — ptej se systému a dostávej odpověď / doptání / upřímné „nevím".

Spuštění:  python3 ask.py      (piš otázky; prázdný řádek nebo Ctrl-D ukončí)

Kontext SE ŘETĚZÍ mezi tahy (`carry_context=True`): navazující otázka bez vlastní entity
(„Kdy se narodil?") si vezme téma z minula („Karel Čapek" ještě svítí), otázka s vlastní
entitou téma přepne. Světlo mezi tahy pohasíná (decay). Runtime je deterministický; jediná
živá služba je UDPipe 2 (rozbor otázky). Odpovědní režim zrcadlí dialogový automat: jasno →
odpověz, nejasno → doptej se, prázdno → nehádej.
"""
from answering import Answering


def main():
    a = Answering()
    print("Ptej se (prázdný řádek ukončí). Např.: Kdo napsal Švejka?")
    while True:
        try:
            q = input("\n? ").strip()
        except EOFError:
            break
        if not q:
            break
        r = a.answer(q, carry_context=True)          # světlo i mount PŘETRVAJÍ mezi tahy
        if not r:
            print("  — nerozumím otázce (chybí tázací slovo, nebo o tom nejsou data)")
        elif r["mode"] == "answer":
            print(f"  → {r['answer']}")
        elif r["mode"] == "clarify":
            nabidka = ", ".join(str(o) for o in r["offer"][:4] if o)
            print(f"  nejsem si jist — mám data o: {nabidka}.  Upřesni prosím?")
        else:
            print("  — o tom nemám jasná data (nehádám)")


if __name__ == "__main__":
    main()
