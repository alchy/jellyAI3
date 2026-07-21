#!/usr/bin/env python3
"""hypothesis-two — rozhraní k Ollama (qwen3.6), think VYPNUTÝ (jinak zdržuje).

Dvě cesty (qwen jde tam i zpět):
  gen_questions(fact, answer, n)  — TAM:  z faktu+odpovědi vyrobí n parafrází otázek
  verify(question, answer)        — ZPĚT: ověří, že odpověď na otázku je věcně správná
"""
import json, urllib.request

MODEL = "qwen3.6:latest"
URL = "http://127.0.0.1:11434/api/generate"

def call(prompt, model=MODEL, timeout=90, temp=0.7):
    data = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "think": False, "options": {"temperature": temp}}).encode()
    req = urllib.request.Request(URL, data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["response"]

def _lines(out):
    res = []
    for l in out.splitlines():
        l = l.strip().lstrip("-•*0123456789.) ").strip().strip('"„""')
        if l:
            res.append(l)
    return res

def gen_questions(fact, answer, n=3, model=MODEL):
    """TAM: n různě formulovaných českých otázek, na které fakt odpovídá 'answer'."""
    p = (f"Fakt: \"{fact}\"\n"
         f"Hledaná odpověď: {answer}\n"
         f"Napiš {n} různě formulovaných českých otázek, na které je podle toho faktu "
         f"odpovědí právě „{answer}\". Použij parafráze — jiný slovosled, synonyma, "
         f"různé tázací vazby. Vrať POUZE otázky, každou na samostatném řádku, "
         f"bez číslování a bez jakéhokoli dalšího textu.")
    qs = [q for q in _lines(call(p, model)) if q.endswith("?")]
    return qs[:n]

def gen_answer(question, fact, model=MODEL):
    """Šablona odpovědi jako CELÁ VĚTA (utvořená z faktu na otázku). Ta věta je
    sama tvrzení = fakt → indexuje se stejně, má vlastní šablonu, lze ji znovu ptát.
    (Fragment odpovědi = answer-slot registru; tady doplňujeme větnou podobu.)"""
    p = (f"Fakt: \"{fact}\"\nOtázka: \"{question}\"\n"
         f"Odpověz na otázku JEDNOU stručnou celou českou větou vycházející z faktu. "
         f"Vrať pouze tu jednu větu, nic dalšího.")
    return call(p, model).strip().strip('"„""')

def verify(question, answer, model=MODEL):
    """ZPĚT: je 'answer' věcně správná odpověď na 'question'? → bool."""
    p = (f"Otázka: \"{question}\"\n"
         f"Navržená odpověď: „{answer}\"\n"
         f"Je ta odpověď věcně přijatelnou odpovědí na tu otázku? "
         f"Odpověz jediným slovem: ANO nebo NE.")
    return call(p, model, temp=0.0).strip().upper().startswith("ANO")

if __name__ == "__main__":     # rychlý self-test
    f = "Mezi pátečníky patřili kromě bratří Čapků mj. prezident T. G. Masaryk."
    qs = gen_questions(f, "prezident Masaryk", 3)
    print("TAM (otázky):")
    for q in qs:
        print("  ", q, " → verify:", verify(q, "prezident Masaryk"))
