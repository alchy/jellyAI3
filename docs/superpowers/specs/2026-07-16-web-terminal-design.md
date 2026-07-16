# Interaktivní web UI — terminál okno + detail uzlu (návrh)

**Datum:** 2026-07-16 · **Větve:** jellyAI3 `feature/web-terminal`, viewBase feature
**Status:** Schváleno (design), čeká na spec-review

## 1. Cíl a kontext

Z živého běhu `./jelly web` vzešly dvě potřeby:
1. **Terminál okno** — vstup i **výstup** v jednom (REPL/konzole), širší než
   `ControlWindow`. Dnes `cmd_web` tiskne odpověď na **stdout serveru**, takže ji
   v prohlížeči nevidíš. Terminál to řeší.
2. **Detail uzlu** — klik na uzel dnes otevře prázdné detail okno. Naplnit ho tím, co
   o uzlu **držíme** (typ, váha, fakty z korpusu). Výukově cenné.

Obojí = „ukaž jellyAI3 info ve viewBase UI". Terminál je nový **typ okna ve viewBase**
(feature do viewBase repa); detail uzlu využije **stávající** `on_click` + `show_detail`.

## 2. Klíčová rozhodnutí

- **Terminál = nový typ okna ve viewBase** (mirror `ControlWindow`, navíc výstup).
  Python API `open_terminal(...)`, protokol delta „terminal_append", frontend komponenta.
- **Odpověď se ukáže v prohlížeči** (terminál), ne na stdoutu.
- **Detail uzlu** přes stávající `on_click(func)` → sestavit řádky z faktů uzlu →
  `detail_window(rows)` / `show_detail`. Žádná nová viewBase feature.
- jellyAI3 zůstává viewBase-free v jádře; vše v adaptéru/`cmd_web` (líný import).

## 3. Terminál okno — viewBase feature

**Python (`viewbase/controls.py` + `canvas.py`):**
```python
class TerminalWindow:                 # jako ControlWindow, ale konzole
    def __init__(self, window_id, *, title="", prompt="❓ ", width=520): ...
    def spec(self) -> dict: ...        # {window_id, title, prompt, width, kind:"terminal"}

# Canvas:
def open_terminal(self, window, *, on_input=None) -> str: ...   # jako open_window
def terminal_write(self, window_id, text) -> None: ...          # push řádku (delta)
```
- `on_input` callback dostane napsaný řádek (event „terminal_input", analogicky
  `window_submit`).
- `terminal_write` zařadí akci/deltu, kterou WS pošle na frontend („terminal_append").

**Protokol (`protocol.py`):** nová akce `{"action":"terminal_append","window_id":…,"text":…}`
v patch deltách; init/patch mechanismus beze změny.

**Frontend (`frontend/src/render/`):** komponenta `terminal_window.js` — širší okno se
scrollovatelným výstupem + input řádkem; při Enter pošle „terminal_input"; na
„terminal_append" připíše řádek a odscrolluje. (Sdílí `base_window.js`.)

## 4. Napojení Q&A (jellyAI3 `cmd_web` + `ViewBaseView`)

```python
# ViewBaseView (adaptér): rozšířit port GraphView o terminál
def open_terminal(self, on_input): ...      # obalí canvas.open_terminal
def write(self, text): ...                  # obalí canvas.terminal_write

# cmd_web:
def on_query(question):
    answer = answerer.answer(question, [])
    view.write(f"❓ {question}\n💬 {answer.text}")   # odpověď v prohlížeči
    reflect(view, answerer)                          # rozsvítí graf + flow
view.open_terminal(on_query)
```
Port `GraphView` získá volitelné `open_terminal(on_input)` + `write(text)` (adaptéry
bez terminálu je nemusí mít — fasáda si poradí).

## 5. Detail uzlu — jellyAI3 info na klik

viewBase `on_click(func)` volá Python callback s uzlem. jellyAI3 sestaví řádky
z grafu a ukáže je:
```python
@canvas.on_click
def show_node(node):
    node_id = node["id"] if isinstance(node, dict) else node
    rows = node_detail_rows(graph, node_id)   # [(klíč, hodnota), …]
    canvas.detail_window(rows); canvas.show_detail(node_id)
```
`node_detail_rows(graph, node_id)` (nová jellyAI3 funkce, `jellyai/viz/detail.py`):
- typ uzlu, váha (frekvence),
- fakty, v nichž vystupuje: „napsat → Babička", „narodit → 1890" (z `facts_of`),
- pro faktový uzel: predikát + účastníci.

Čistě z grafu — testovatelné bez viewBase (`FakeView`/přímý dict).

## 6. Separace a lifecycle

- Jádro `import jellyai` **nesáhne** na viewBase (drží se; terminál/detail jen v adaptéru).
- viewBase vlastní webserver lifecycle (`serve`/`stop`) — beze změny.
- `node_detail_rows` je viewBase-free (jen graf) → hermeticky testovatelné.

## 7. Testy

- **node_detail_rows** (hermeticky): pro uzel „Babička" vrátí řádky s „napsat" a
  „Božena Němcová"; pro faktový uzel predikát + účastníky.
- **cmd_web s FakeView** (rozšířeným o `open_terminal`/`write`): po `on_query`
  se zavolá `view.write` s odpovědí a `reflect`.
- **GraphView port**: `open_terminal`/`write` volitelné (fasáda funguje i bez nich).
- viewBase strana: Python `TerminalWindow.spec()` má `kind:"terminal"`; `terminal_write`
  zařadí správnou akci (unit test ve viewBase repu).

## 8. Rozsah / fáze

- **A (jellyAI3, bez viewBase změn):** `node_detail_rows` + napojení `on_click` v
  `cmd_web` (detail uzlu funguje hned přes stávající viewBase).
- **B (viewBase feature):** `TerminalWindow` (Python + protokol + frontend + rebuild
  + commit static) → `cmd_web` napojí Q&A výstup do prohlížeče.
- **Mimo:** stylování terminálu, historie příkazů (šipky), více terminálů.
