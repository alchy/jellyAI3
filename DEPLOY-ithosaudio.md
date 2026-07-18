# Nasazení jellyAI3 — jelly.ithosaudio.eu

Produkční nasazení na `mail1.lordaudio.eu` (116.203.157.235). Vzor: `learn.ithosaudio.eu`.
Zdroj: https://github.com/alchy/jellyAI3 + komponenta https://github.com/alchy/viewBase

## Přehled

- **Web (viewBase):** `127.0.0.1:8080` ← nginx `jelly.conf` (HTTPS + WebSocket + basic auth).
- **Iris REST:** `127.0.0.1:8084` (child procesu web; startuje líně při 1. dotazu).
- **ÚFAL služby (child procesy):** nametag `8091`, udpipe `8092`, morpho `8093`.
- **systemd:** `jelly.service` (User=`jelly`, `./cli.py web`), enabled. Celý strom
  (web + iris + ÚFAL služby) je v jednom cgroupu → restart/kill obslouží systemd.
- **nginx:** `/etc/nginx/conf.d/jelly.conf`, basic auth `ithosaudio` (soubor
  `/etc/nginx/conf.d/.htpasswd.jelly`, apr1). Cert LE `jelly.ithosaudio.eu` (ECDSA, webroot).

## Úpravy proti upstreamu (NUTNÉ pro tento stroj)

1. **`config.py` — přemapované porty ÚFAL služeb.** Výchozí 8081/8082/8083 kolidují
   s běžícími uniforms/cudlik/learn → `nametag_port=8091, udpipe_port=8092, morpho_port=8093`.
   web(8080) a iris(8084) byly volné, ponechány.
2. **`jellyai/ufal_client.py` — `UfalClient._ensure` health-first.** Web drží morpho
   (popisky uzlů) a Iris ji chce taky (skloňování odpovědí) → bez health-checku by
   druhý proces nesbindoval port a spadl. Teď běžící službu na portu SDÍLÍ
   (jako `IrisClient`). `close()` přeskakuje cizí (None) handly.
3. **`cli.py` — `view.serve(open_browser=False)`** (server, žádný lokální prohlížeč).
4. **`jelly` launcher — `python3` místo `python3.11`** (v systému je jen 3.12; `requires-python>=3.11` OK).
5. **`services/iris_service.py` — e-mailový kanál Chronos připomínek.** V `_channels()`
   přidán kanál `email` (rozšiřovací bod, který autor v kódu předpokládal). Když dozraje
   připomínka, `ChronosTicker`→`notify` ji pošle VŠEM kanálům včetně e-mailu.

## Připomínky e-mailem (Chronos)

- Kanál `email` posílá text dozrálé připomínky přes lokální Postfix (`127.0.0.1:25`).
- Řízeno env proměnnými v `jelly.service`:
  - `JELLY_REMINDER_EMAIL=jindrich.nemec@yahoo.com` (adresát; bez něj je kanál vypnutý)
  - `JELLY_REMINDER_FROM=jelly@lordaudio.eu` (**MUSÍ být @lordaudio.eu** — OpenDKIM
    podepisuje `*@lordaudio.eu` selektorem `mail`, SPF alignment → Yahoo přijímá `dirdel`).
- Test: nastav připomínku v okně/konzoli („připomeň mi za minutu …"), po dozrání přijde
  mail. Ověřeno: `from=jelly@lordaudio.eu`, DKIM `s=mail d=lordaudio.eu`, `status=sent`.
- Změna adresáta: uprav `Environment=` v `/etc/systemd/system/jelly.service` + `daemon-reload`
  + `systemctl restart jelly.service`.

### Pojmenovaný adresát (kontakty přes Mnemos)

- **Uložení e-mailu osoby:** „zapamatuj si že Jindra má email jindrich.nemec@yahoo.com".
  E-mail je HODNOTA MIMO přirozený jazyk → rozpozná ho **regexp** (`EMAIL_RE`
  v `mnemos.py`), který obejde tokenizér i tagger; uloží se jako **typovaný fakt
  grafu** `email(osoba, adresa)` s participantem typu `email` (přes kartu
  `patterns/cs/statement-attribute.json`, rys `email` má nejvyšší specificitu).
  Fakt jde do deníku `data/memory.jsonl` → přežije restart (replay).
- **Adresované připomínky:** „pošli Jindrovi zítra ráno upozornění XYZ" — adresát
  (jméno za „pošli", shoda přes kmen napříč pády Jindrovi→Jindra) se dohledá
  z grafu a uloží do `record["recipient"]`. `fire_due` ho nese v `ReminderMessage.recipient`,
  e-mailový kanál pošle tam. **Bez pojmenovaného adresáta → default** (env).
  Neznámé jméno → Iris upřímně vyzve k doplnění kontaktu.
- Změny: `mnemos.py` (EMAIL_RE, name_stem, rys/typ email, kind=attribute),
  `automaton.py` (ReminderMessage, _send_command, _resolve_person_email),
  karta `statement-attribute.json`, cs.json (`send_phrases`, `recipient_pronouns`).

## Chyták: titulky uzlů (CSP × font CDN)

- viewBase kreslí popisky uzlů přes **troika-three-text** (SDF text), která si
  glyfy fontu tahá přes `fetch()` z **`https://cdn.jsdelivr.net`** (`unicode-font-resolver`).
- Přísná CSP `connect-src 'self'` ten fetch zablokuje → text bez fontu = **neviditelné
  titulky** (ve všech prohlížečích, ne klientský stav!). V `jelly.conf` proto CSP
  povoluje `https://cdn.jsdelivr.net` v `connect-src` i `font-src`.
- Runtime závislost na jsDelivr. Pro plně offline nasazení by šlo font zabalit do
  `viewbase/static` a resolver přesměrovat (větší zásah, zatím neuděláno).

## Časové pásmo

- Chronos plánuje připomínky v **Europe/Prague** (uživatel je v Praze). Systém byl
  původně UTC → `timedatectl set-timezone Europe/Prague` + `Environment=TZ=Europe/Prague`
  v `jelly.service` (pojistka pro plánování i kdyby se systém vrátil na UTC).

## Rebuild od nuly

```bash
cd /www/jelly
./jelly setup            # venv (3.12) + numpy; pak doinstaluj ufal + viewbase:
.venv/bin/pip install 'ufal.morphodita>=1.11' 'ufal.nametag>=1.2' 'ufal.udpipe>=1.3' ./_viewBase/python websockets
./jelly qa-models        # stáhne ÚFAL modely do data/models (~96 MB, LINDAT)
./jelly prepare          # seed R.U.R. + index (data/index.pkl)
./jelly annotate         # anotace vět (spustí nametag+udpipe) → data/annotations.pkl
./jelly graph            # faktový graf → data/graph.pkl
chown -R jelly:jelly /www/jelly
systemctl restart jelly.service
```

## Restart / provoz

- **Restart:** `systemctl restart jelly.service` (iris/ÚFAL child procesy se složí a
  nastartují znovu líně). Frontend (viewBase) je no-cache → změny assetů bez restartu.
- **Rebuild grafu za běhu:** přehraj pipeline (`annotate`/`graph`) a `systemctl restart jelly.service`.
- **Log:** `journalctl -u jelly.service -f` (odpovědi na dotazy se logují).
- **Ruční test dotazu (bez prohlížeče):**
  `curl -s -X POST http://127.0.0.1:8080/api/event -H 'Content-Type: application/json' -d '{"event":"terminal_input","payload":{"window_id":"konzole","line":"Kdo napsal R.U.R.?"}}'`
  → odpověď se objeví v journalu (a v otevřených oknech prohlížeče).

## Ověření (E2E)

```bash
curl -sI https://jelly.ithosaudio.eu/                    # 401 (bez auth)
curl -sI -u ithosaudio:iris https://jelly.ithosaudio.eu/ # 200 (index.html)
# WebSocket přes nginx viz README repo / test v historii nasazení
```

> `_viewBase/` = zdroj komponenty viewBase (má i předpřipravený frontend
> `python/viewbase/static/` → npm build netřeba). Instalováno do venv přes pip.
