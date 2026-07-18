# infra — nasazení jellyAI3 (systemd + nginx)

Příklady infrastruktury pro produkční běh (podle reálného nasazení
`jelly.ithosaudio.eu`). Uprav cesty, doménu, uživatele a e-maily podle sebe.

## Obsah

| Soubor | Popis |
|---|---|
| `jelly.service` | systemd unit — spustí `cli.py web` pod uživatelem `jelly` (hardening, e-mailový kanál, TZ). |
| `jelly.conf` | nginx vhost — HTTPS + **WebSocket** proxy na `127.0.0.1:8080`, basic auth, CSP (povoluje `cdn.jsdelivr.net` kvůli fontu popisků). |

## systemd

```bash
# uživatel + adresář aplikace (kód v /www/jelly, vlastník jelly:jelly)
useradd --system --home-dir /www/jelly --shell /usr/sbin/nologin jelly

cp infra/jelly.service /etc/systemd/system/jelly.service
#   → uprav Environment (JELLY_REMINDER_EMAIL/FROM, TZ), případně porty v config.py
systemctl daemon-reload
systemctl enable --now jelly.service
systemctl status jelly.service
journalctl -u jelly.service -f          # log (odpovědi na dotazy)
```

Restart po změně: `systemctl restart jelly.service` (Iris/ÚFAL child procesy se
složí a nastartují znovu líně; frontend viewBase je no-cache → assety bez restartu).

## nginx + TLS + basic auth

```bash
mkdir -p /var/log/nginx/jelly
# basic auth (uživatel + heslo)
printf 'user:%s\n' "$(openssl passwd -apr1 HESLO)" > /etc/nginx/conf.d/.htpasswd.jelly
chown root:nginx /etc/nginx/conf.d/.htpasswd.jelly && chmod 640 /etc/nginx/conf.d/.htpasswd.jelly

cp infra/jelly.conf /etc/nginx/conf.d/jelly.conf     # uprav server_name + cesty k certu
# cert (Let's Encrypt, webroot): nejdřív port-80 blok pro ACME, pak:
certbot certonly --webroot -w /var/www/letsencrypt -d TVOJE.DOMENA --key-type ecdsa
nginx -t && systemctl reload nginx
```

> **Pozn.:** CSP v `jelly.conf` povoluje `https://cdn.jsdelivr.net` — troika-three-text
> (SDF popisky uzlů) si odtud tahá font. Bez toho jsou titulky uzlů neviditelné.
> Alternativa: font zabalit lokálně (BACKLOG #36) a CSP nechat přísnou.
