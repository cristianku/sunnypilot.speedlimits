# API REST e server MCP della revisione cartelli

Il sito di revisione (`nnslr review`, servizio `speedlimits-web`) si può usare anche senza browser:

- **REST API** (`/api/v1/...`) per qualsiasi client HTTP: `curl`, uno script Python, un altro programma;
- **server MCP** per Claude Code, Codex e gli altri client [Model Context Protocol](https://modelcontextprotocol.io), con due trasporti: **Streamable HTTP** su `/mcp` dello stesso servizio e **stdio** (`helpers/mcp.sh`, cioè `nnslr mcp`).

Con REST e MCP il server non contiene nessun modello e nessun agente: l'intelligenza artificiale gira sul computer del client, e il server dà soltanto strumenti per leggere fotogrammi e immagini, consultare proposte e revisioni e salvare conferme o correzioni. Il sito ha in più un revisore AI proprio, che chiama un server compatibile con l'API OpenAI scelto in Settings: si usa solo dal sito, non da REST o MCP (sezione 8).

Le regole per **interpretare** i cartelli restano in [review_instructions.md](../review_instructions.md) (sezioni 4–5); qui c'è come si usano le interfacce.

## 1. Architettura

```
                    ┌──────────────── speedlimits-web (nnslr review, ThreadingHTTPServer) ───────────────┐
browser ──────────► │ pagina /route/..., /api/route/...  (cookie col token, come prima)                 │
curl, Python ─────► │ /api/v1/...        (header Authorization: Bearer)  ─┐                             │
Claude Code (HTTP)► │ /mcp  (header Authorization: Bearer) → SDK MCP ─────┤                             │
                    └─────────────────────────────────────────────────────┼─────────────────────────────┘
Claude Code (stdio, anche via ssh) → helpers/mcp.sh → SDK MCP ─────────────┤
                                                                           ▼
                        nnslr_tools.review_service (ReviewService: route, fotogrammi, immagini, salvataggi)
                                                                           ▼
                        nnslr_tools.review_db.save_frame_review()  →  data/annotations/review.sqlite
                        (stesse validazioni dell'import, cronologia append-only, revisioni)
```

- `review_service.py` è il nucleo condiviso: ordine dei fotogrammi, filtri, sospetti, immagini, salvataggio. Usa solo la libreria standard (Pillow, se c'è, solo per ritagli e ridimensionamenti).
- `review.py` (il sito) serve la pagina, la REST API e `/mcp`; `review_mcp.py` definisce gli strumenti MCP con l'SDK ufficiale `mcp` (2.x).
- Pagina, REST e MCP passano tutti da `save_frame_review()`: stesse validazioni, stesso database, stesso controllo delle revisioni.
- L'SDK MCP è una **dipendenza opzionale**: senza, il sito, la pagina e `/api/v1` funzionano; `/mcp` risponde 501 e `nnslr mcp` esce con un messaggio.
- Il pannello (`/pannello`: prepara, importa, allena, aggiorna) **non** è esposto né in `/api/v1` né in MCP. Nemmeno il revisore AI del sito (sezione 8).

## 2. Autenticazione e sicurezza

- `/api/v1/...` e `/mcp` vogliono **sempre** il token, anche quando il sito ascolta solo su 127.0.0.1 (dove la pagina non lo chiede). Il token è `NNSLR_REVIEW_TOKEN`: sul server è nel `.env` del repository e in `/etc/speedlimits-web.env` (vedi [GUIDE.md](GUIDE.md), sezione 3). Su 127.0.0.1 senza `NNSLR_REVIEW_TOKEN` il server ne genera uno a caso e lo stampa all'avvio.
- Il token va **in un header**: `Authorization: Bearer <token>` (oppure `X-Review-Token: <token>`). Il cookie della pagina non vale per l'API. Un `?token=` nell'indirizzo è rifiutato con 400 `token_in_url` (finirebbe in log e cronologie).
- Senza token o con un token sbagliato: 401 `unauthorized` con `WWW-Authenticate: Bearer`.
- Un header `Origin` diverso dal server stesso (o `Sec-Fetch-Site: cross-site`) → 403 `cross_origin`: un sito web aperto nel browser non può chiamare l'API o `/mcp` (protezione DNS rebinding richiesta dal trasporto MCP). I client da terminale non mandano `Origin`.
- Sul sito in sola LAN locale (127.0.0.1) resta anche il controllo dell'header `Host`.
- Corpo massimo 512 KiB (413), solo JSON (415), al massimo 100 riquadri per fotogramma.
- Le immagini: l'anteprima è solo `images/NNNNNNNN.jpg` del report corrente; l'originale è solo il PNG estratto sotto `data/derived/frames/<route>/`, mai un link simbolico, mai un percorso con `..`, e con lo SHA-256 del report (409 `image_changed` se il file è cambiato).
- Reviewers from outside ([GUIDE.md](GUIDE.md), "People from outside") never reach `/api/v1` or `/mcp`: through the proxy both answer 403, with or without the token.
- Il server non scrive mai il token nei suoi messaggi. Non mettere il token in file versionati, resoconti o log: negli esempi sotto è sempre la variabile d'ambiente `NNSLR_REVIEW_TOKEN`.

## 3. Provenienza: revisori umani e AI

Ogni revisione salvata registra:

| Campo | Valori | Dove |
| --- | --- | --- |
| `reviewer` | nome o sigla; per un'AI è salvato come `ai/<nome>` (es. `ai/Claude Code`) | DB, payload, cronologia |
| `reviewer_kind` | `human` o `ai` | DB e cronologia (colonna nuova); per le AI anche payload e `provenance` del dataset |
| `channel` | `web` (pagina), `rest` (`/api/v1`), `mcp` | DB e cronologia (colonna nuova); per le AI anche payload (`review_channel`) e `provenance.review_channel` del dataset |
| `label_basis` | `human_review` per le persone, `ai_review` per le AI | payload salvato e `provenance` del dataset |

- **MCP salva sempre come `ai`**: il nome dato dal client diventa `ai/<nome>`, anche se fosse `cristianku`.
- La **REST API** vuole `reviewer_kind` esplicito (`human` o `ai`). Un nome che inizia con `ai/` con `reviewer_kind: "human"` è rifiutato.
- La pagina continua a salvare come `human` (canale `web`), come prima. Un agente che usa il browser resta riconoscibile solo dal nome che scrive nel campo Revisore: per le AI è meglio usare MCP o REST. Le revisioni salvate così da un'AI si riclassificano con `nnslr review-reclassify-ai --reviewer Codex --reviewer ChatGPT` (prima `--dry-run`): ogni fotogramma riceve una nuova revisione `ai/<nome>`, tipo `ai`, `label_basis` `ai_review`, con gli stessi riquadri e la stessa ora di revisione, e la cronologia lo registra. Così compaiono nel filtro **AI** della pagina ed entrano nel dataset come revisioni AI, non come umane.
- Il **revisore AI del sito** (sezione 8) salva solo da **Review and save** (`ai_review` con `save: "yes"`), come `ai/<provider>/<modello>`, tipo `ai`, canale `web`. Un suggerimento chiesto dalla pagina non salva niente: se la persona lo applica e conferma, è la sua revisione (`human`, `web`).
- Un revisore AI **non modifica e non ritira** un fotogramma confermato da una persona: 409 `human_review_exists`, a meno di `override_human_review: true`, da usare solo se Cristian lo chiede per quel fotogramma.
- **Una conferma AI è definitiva**: nessuna persona deve riapprovarla. `nnslr import-annotations --from-db` (cioè `helpers/3_train.sh` o **Importa** nel pannello) importa i fotogrammi confermati da persone e da AI, anche quelli salvati prima di questa regola. Nel dataset le righe AI restano riconoscibili: `reviewer` `ai/<nome>`, `provenance.label_basis` `ai_review`, `provenance.reviewer_kind` `ai`, `provenance.review_channel` `rest`, `mcp` (o `web` per quelle riclassificate); un riquadro aggiunto dall'AI ha `proposal_source` `ai_review`. Il report dell'import conta i fotogrammi AI nuovi in `imported_ai_frame_count`. Le righe umane restano identiche a prima.
- Non entrano mai nel dataset i fotogrammi mai confermati, le conferme ritirate (`confirmed: false`) e le proposte del modello. Le revisioni automatiche `model_review` restano solo `training-candidate`.
- Il dataset segue l'ultima revisione: al prossimo import un fotogramma già importato e poi modificato (o riclassificato come AI) sostituisce le sue righe vecchie (`replaced_frame_count`), uno già importato e poi ritirato viene tolto (`removed_withdrawn_frame_count`). Le righe tolte restano archiviate in `data/annotations/replaced/`.
- L'export JSONL della pagina include anche le conferme AI, con `label_basis` `ai_review`, `reviewer_kind` e `review_channel`.
- Confermare non avvia niente: import e training restano comandi separati ed espliciti.
- Il database passa da solo dalla versione 1 alla 2 alla prima apertura: le due colonne nuove vengono aggiunte e le revisioni già salvate (tutte dalla pagina) risultano `human`/`web`. Una versione vecchia di `nnslr` rifiuta poi di aprire il database (versione 2): aggiornare il codice prima di riavviare il servizio.

## 4. REST API

Base: `http://<server>:8765/api/v1`. Tutte le risposte sono JSON (tranne le immagini), con `Cache-Control: no-store`. Gli errori hanno sempre la forma:

```json
{"ok": false, "error": "revision_conflict", "message": "Questo fotogramma è stato salvato da un’altra pagina nel frattempo: ..."}
```

| Metodo e percorso | Cosa fa |
| --- | --- |
| `GET /api/v1` | Versione dell'API e se MCP è disponibile |
| `GET /api/v1/schema` | Campi e valori ammessi per salvare, regole, codici di errore |
| `GET /api/v1/routes` | Route con report: `run_id`, fotogrammi, confermati (umani/AI), mancanti, ritirati, sospetti, con proposte, con seconda lettura, e `drive`: il viaggio come lo chiama il sito (`title` "Herznach → Zürich", `when` "Fri 23 Oct 2026 · 08:01", `minutes`, `start`, `from`, `to`; `null` quel che non si sa; `full_definition` true for a drive recorded at 2688 × 1520); `dashcam`: true for a dashcam drive (sunnypilot off in all of it, the brain's), false with sunnypilot on, null before its logs are read |
| `GET /api/v1/routes/{route}` | Avanzamento: gli stessi conteggi e `drive`, i sospetti con i motivi (`suspicious_limit`, default 100), i confermati con dubbi, il primo da rivedere |
| `GET /api/v1/routes/{route}/frames?filter=&limit=&cursor=` | Fotogrammi in ordine (segmento, immagine), a pagine. `filter`: `unreviewed` (default), `reviewed`, `suspicious`, `with_proposals`, `all`; `limit` 1–200 (default 50); `cursor` = `next_cursor` della pagina prima |
| `GET /api/v1/routes/{route}/next?filter=&after=` | Il prossimo fotogramma da fare, con tutto il dettaglio; `after` = chiave da cui proseguire (ricomincia dall'inizio una volta, `wrapped: true`) |
| `GET /api/v1/routes/{route}/frames/{seg}/{img}` | Dettaglio di un fotogramma |
| `GET /api/v1/routes/{route}/frames/{seg}/{img}/context?before=&after=` | I vicini (0–10 per parte, default 2) nell'ordine del report, anche oltre il confine del segmento |
| `GET /api/v1/routes/{route}/frames/{seg}/{img}/image?variant=&crop=&long_side=` | L'immagine: `variant=preview` (JPEG del report, default) o `original` (PNG estratto); `crop=x1,y1,x2,y2` in pixel originali e `long_side=32..2048` richiedono Pillow sul server. On a drive at full definition `width`, `height` and the boxes stay 1344 × 760, while `preview` is the frame as recorded, 2688 × 1520; `crop` keeps the frame's pixels of the row |
| `GET /api/v1/routes/{route}/frames/{seg}/{img}/history` | Cronologia: ogni revisione con revisore, tipo, canale, ora e riquadri |
| `POST /api/v1/routes/{route}/frames/{seg}/{img}/review` | Salva conferma, correzione o ritiro |
| `GET /api/v1/sign-distance` | Data › Sign distance, read only: every drive with its state (`signs`, `unread`, `unreadable`, `no_logs`, `build_too_old`, `no_sign_seen`, `no_video`, `reading`) and its counts, and `summary` (the median `late_s` with its count, smallest and largest, `late_m` with its `note`, the signs left out by reason, the marks out of the medians, the orphan marks, `by_drive`) |
| `GET /api/v1/sign-distance/{route}` | One drive of Sign distance, read only: its state, its signs (vision's zero `zero_s`, the last sighting, `included` or `excluded` with the reason, the `mark` in effect and its `metrics`: `late_s`, `late_m`, `vision_m`, `v_kmh`) and its `orphans` |

Gli endpoint della pagina (`GET /api/routes`, `GET /api/route/{route}/reviews`, `POST /api/route/{route}/frame/{seg}/{img}`) restano identici, con cookie o `X-Review-Token` come prima. `GET /api/route/{route}/reviews` ora indica anche `reviewer_kind` e `channel` di ogni revisione.

Every frame of the video (the page only): `GET /api/route/{route}/video/{seg}/{frame}` decodes the GOP of that frame of the segment's video (20 a second; it answers once the GOP's first frame is decoded, the rest following in the background, and 404 `no_video_frame` on a GOP it cannot decode) and gives `frame_count`, `gop` (first and last frame), `seconds` (the seconds a segment kept only at the signs has, else `null`) and `frames`, each with its `frame_key`: the key of the picture of the report when there is one, else `{seg}/{1000 + frame}`; `GET /route/{route}/video/{seg}/{frame}.jpg` is its picture. `POST /api/route/{route}/frame/{seg}/{1000 + frame}` with `confirmed: true` takes such a frame into the report (its own PNG, `video_frames.jsonl`) and saves the review, with its boxes or with none (no sign); `.../apply-previous` and `.../fit` take such a frame too (they save nothing), with the frame of the report just before it as the previous one. The page's payload has `videos`: for every segment whose road video is on the server, its frame count (`null` until a frame of its video is saved). The REST API and MCP save only frames of the report, and `next`/`next_frame` never walk the frames of the video that are not saved.

What the car's networks read on a frame (the page only, **Car networks**, key T): `GET /api/route/{route}/car/{seg}/{img}` runs the car's SignNet and classifier (`vision.read_signs` and the release of the sunnypilot clone) on that frame, a picture of the report or a frame of the video `{seg}/{1000 + frame}`, and gives `signs` (each with `box` in the frame's pixels, `score`, `class`, `confidence`, `speed`, `use`: `speed`, `road`, `board`, `nothing` or `unsure`, `review`, the class as a box of the review, and, but on a box read as not a sign, `applicability` and `applicability_score`, from the applicability network in use), `applicability_note` (why that network gave no answer, else `null`), `surest` (the index of the reading the car's logs would show, or `null`), `width`, `height`, `release` and `ms`. It saves nothing. 404 `no_video_frame` for a frame without its road video, 503 `no_car_networks` when the clone has no release. The REST API and MCP do not have it.

What the classifier gets of a box (the page only, the square beside Sign review's zoom): `GET /api/route/{route}/frame/{seg}/{img}/classifier-crop?box=x1,y1,x2,y2[&samples=0..8]` answers a PNG strip of `input_size` squares, made by the training's own cuts (`crop_cache.review_crops`). First comes the crop without jitter, cut from the frame at full definition when there is one; then `samples` crops of the box jittered as the augmented training jitters it, without the rotation or the colours. The box is in the frame's pixels (1344 × 760 on every drive), saved or not. Headers: `X-Crop-Source` (the picture it was cut from, e.g. `2688x1520`), `X-Crop-Pixels` (the box's pixels in it), `X-Crop-Size` and `X-Crop-Samples`. A guest with the drive's signs may read it. Errors: 400 `invalid_box`/`invalid_samples`; 404 `unknown_route`/`unknown_frame`/`image_not_found`; 409 `image_changed`/`invalid_frame`/`invalid_image`. It saves nothing. The REST API and MCP do not have it.

Sign distance (Data › Sign distance: how late the comma's countdown to the sign ahead reaches zero, against the frame a person marks as zero) is **read only** for the REST API: `GET /api/v1/sign-distance` and `GET /api/v1/sign-distance/{route}` above, with the same Bearer token. `late_s` is the main figure (vision's zero minus the person's, in seconds: + late, − early); `late_m` is secondary and includes speedvisiond's processing delay times the speed. **An agent never sets a zero**: the marks are made only by a person on the site (the page's `POST /api/sign-distance/drives/{route}/marks` and `.../undo`): signed in, with the session cookie and never the token alone, or the person at the server itself on a site served only on 127.0.0.1 without sign-in (reviewer `local`). There is no REST write and no MCP tool for them: a `POST` under `/api/v1/sign-distance` answers 404. The marks are kept in `data/annotations/sign_distance/<route>.jsonl`, a file only ever added to.

Speed review (Data › Speed review: the blocks that correct the driver's own speed, the brain's answer, from a time to a time of a drive, the roadworks and the kinds of road marked on it, and the kinds of road themselves) is **not** in the REST API or in MCP: only a person signed in on the site writes them, and the token alone is refused.

### Ordine, cursori e ripresa

- L'ordine è quello della timeline della pagina: per segmento e poi per immagine. Gli indici non devono essere consecutivi (`0/0, 0/2, 0/5, 1/1, ...`): precedente e successivo sono i vicini in quest'ordine, anche tra segmenti diversi; `same_segment` dice se un vicino è nello stesso segmento.
- Il cursore è opaco e contiene `run_id`, filtro e ultima chiave: una conferma fatta tra due pagine non sposta il resto. Con un report rigenerato il cursore dà 409 `stale_report`.
- `next` con `filter=unreviewed` e senza `after` riparte sempre dal primo fotogramma non confermato, chiunque abbia confermato gli altri: dopo un'interruzione basta richiamarlo. Non ci sono lock: due revisori sullo stesso fotogramma si accorgono del conflitto al salvataggio (409).

### Dettaglio di un fotogramma (esempio ridotto)

```json
{
  "frame_key": "0/5", "segment_index": 0, "output_index": 5, "position": 2, "frame_count": 6,
  "route_id": "00000001--abcdef1234", "run_id": "20261021T100000Z-0123abcd",
  "status": "unreviewed", "revision": 0, "reviewer": null, "reviewer_kind": null,
  "width": 1344, "height": 760,
  "proposal_count": 1, "second_reading_count": 0, "suspicious": false, "suspicious_reasons": [],
  "proposals": [{"detection_id": "0/5/0", "bbox_xyxy": [1, 2, 15, 20], "family": "maximum_speed",
                 "value_kph": 30, "score": 0.8, "applicability": "unresolved", "...": "..."}],
  "second_reading": [],
  "review": null,
  "previous_frame_key": "0/2", "next_frame_key": "1/1",
  "image": {"preview_url": "/api/v1/routes/00000001--abcdef1234/frames/0/5/image?variant=preview",
            "original_url": "/api/v1/routes/00000001--abcdef1234/frames/0/5/image?variant=original", "...": "..."},
  "metadata": {"frame_sha256": "...", "camera_stream": "narrow_road", "model": {"...": "..."}, "...": "..."},
  "save": {"run_id": "20261021T100000Z-0123abcd", "base_revision": 0}
}
```

`status` è `unreviewed`, `confirmed` o `withdrawn` (conferma ritirata). `save` contiene esattamente i due valori da rimandare al salvataggio.

### Salvare

Corpo del `POST .../review`:

| Campo | Obbligatorio | Significato |
| --- | --- | --- |
| `run_id` | sì | quello letto col fotogramma; diverso → 409 `stale_report`, assente → 400 `missing_run_id` |
| `base_revision` | sì | la `revision` letta (0 se mai salvato); diversa → 409 `revision_conflict`, assente → 428 `base_revision_required` |
| `reviewer` | sì | nome (max 100 caratteri, senza caratteri di controllo) |
| `reviewer_kind` | sì | `human` o `ai` |
| `confirmed` | sì | `true` salva i riquadri; `false` ritira la conferma |
| `detections` | con `confirmed: true` | tutti i riquadri da tenere; `[]` = nessun cartello pertinente |
| `site_group`, `encounter_id` | no | stringa o `null` |
| `override_human_review` | no | vedi sezione 3 |

Ogni riquadro:

- per **tenere o correggere una proposta** basta `detection_id` più i campi che cambiano: gli altri sono copiati dalla proposta del report (per esempio `{"detection_id": "0/5/0", "applicability": "own_road"}`);
- una proposta **non inclusa** viene registrata come scartata (falso positivo);
- un **riquadro nuovo** non ha `detection_id` (il server assegna `manual-rest-1`, `manual-mcp-1`, ...) e ha `bbox_xyxy`, `family`, `value_kph`;
- valori ammessi: `GET /api/v1/schema` e la tabella in [review_instructions.md](../review_instructions.md) sezione 9. `value_kph` è un intero 5–250 o `null` (numero non leggibile: mai inventarlo).

Risposta:

```json
{"ok": true, "status": "confirmed", "confirmed_count": 12, "missing_count": 40,
 "next_unreviewed_frame_key": "1/1",
 "review": {"frame_key": "0/5", "revision": 1, "confirmed": true, "reviewer": "ai/Claude",
            "reviewer_kind": "ai", "channel": "rest", "detections": ["..."], "...": "..."}}
```

### Esempi `curl`

Il token resta nella variabile d'ambiente (sul server: `set -a; . ./.env; set +a`; altrove copiarlo a mano, senza salvarlo in file versionati):

```sh
export NNSLR_REVIEW_TOKEN=...            # non scriverlo in file versionati
BASE=http://<IP del server>:8765/api/v1  # oppure http://127.0.0.1:8765/api/v1 sul server
AUTH="Authorization: Bearer $NNSLR_REVIEW_TOKEN"
ROUTE=0000011e--0c8825628a

curl -sS -H "$AUTH" "$BASE/routes"
curl -sS -H "$AUTH" "$BASE/routes/$ROUTE"                                   # avanzamento e sospetti
curl -sS -H "$AUTH" "$BASE/routes/$ROUTE/frames?filter=suspicious&limit=20"
curl -sS -H "$AUTH" "$BASE/routes/$ROUTE/next"                              # prossimo da rivedere
curl -sS -H "$AUTH" "$BASE/routes/$ROUTE/frames/3/28/context?before=3&after=3"
curl -sS -H "$AUTH" -o frame.jpg "$BASE/routes/$ROUTE/frames/3/28/image"
curl -sS -H "$AUTH" -o sign.png "$BASE/routes/$ROUTE/frames/3/28/image?variant=original&crop=600,200,700,300&long_side=512"

# Conferma: la proposta 3/28/0 vale 60 sulla nostra strada; nessun altro cartello.
curl -sS -H "$AUTH" -H 'Content-Type: application/json' -X POST "$BASE/routes/$ROUTE/frames/3/28/review" -d '{
  "run_id": "20261021T083422Z-2468e469", "base_revision": 0,
  "reviewer": "Cristian", "reviewer_kind": "human", "confirmed": true,
  "detections": [{"detection_id": "3/28/0", "value_kph": 60, "applicability": "own_road"}]}'

# Nessun cartello pertinente (le proposte diventano falsi positivi):
curl -sS -H "$AUTH" -H 'Content-Type: application/json' -X POST "$BASE/routes/$ROUTE/frames/3/29/review" -d '{
  "run_id": "20261021T083422Z-2468e469", "base_revision": 0,
  "reviewer": "Claude", "reviewer_kind": "ai", "confirmed": true, "detections": []}'

curl -sS -H "$AUTH" "$BASE/routes/$ROUTE/frames/3/28/history"
```

### Esempio Python (solo libreria standard)

```python
import json, os, urllib.request

BASE = 'http://127.0.0.1:8765/api/v1'
HEADERS = {'Authorization': 'Bearer ' + os.environ['NNSLR_REVIEW_TOKEN']}

def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(BASE + path, data=data, method=method,
                                     headers={**HEADERS, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request) as response:
        return json.load(response)

route = call('GET', '/routes')['routes'][0]['route_id']
frame = call('GET', f'/routes/{route}/next')['frame']
# ... guardare l'immagine: GET /routes/{route}/frames/{frame_key}/image ...
saved = call('POST', f"/routes/{route}/frames/{frame['frame_key']}/review", {
    **frame['save'], 'reviewer': 'Cristian', 'reviewer_kind': 'human', 'confirmed': True,
    'detections': [{'detection_id': p['detection_id'], 'applicability': 'own_road'} for p in frame['proposals']]})
```

`urllib` solleva `HTTPError` per 4xx/5xx: il corpo dell'errore è il JSON con `error` e `message`.

## 5. Server MCP

### Strumenti

| Strumento | Cosa fa | Scrive |
| --- | --- | --- |
| `list_routes` | Route e avanzamento, con il viaggio (`drive`: titolo, data e ora, minuti) | no |
| `list_frames` | Fotogrammi con filtri e cursore, senza immagini | no |
| `next_frame` | Prossimo fotogramma da rivedere (ripresa dopo interruzione, `after`) | no |
| `get_frame` | Metadati, dimensioni, proposte, seconda lettura, sospetti, revisione salvata, `run_id`, `revision` | no |
| `get_frame_image` | **Contenuto immagine MCP** (`type: image`, base64) più una riga JSON; anteprima JPEG o originale PNG, ritaglio e ingrandimento | no |
| `get_frame_context` | Vicini prima e dopo; con `include_images` anche le immagini dei 4 più vicini | no |
| `get_review_schema` | Campi, valori ammessi, regole, errori | no |
| `save_frame_review` | Conferma, correzione, nessun cartello o ritiro (`confirmed: false`) | **sì** |
| `get_review_progress` | Conteggi, sospetti con motivi, confermati con dubbi | no |
| `get_review_history` | Cronologia del fotogramma | no |

- Tutti gli strumenti hanno schema di input; quelli che restituiscono JSON hanno anche lo schema di output (`structuredContent`). Gli strumenti di lettura sono marcati `readOnlyHint`.
- Gli elenchi non contengono immagini: il client chiede `get_frame_image` solo quando serve, e `get_frame_context` allega al massimo 4 immagini.
- Gli errori arrivano come risultato con `isError: true` e testo `Error executing tool <nome>: {"error": ..., "status": ..., "message": ...}`, con gli stessi codici della REST API.
- `save_frame_review` salva sempre come revisore AI (sezione 3). Con `NNSLR_MCP_READ_ONLY=1` (servizio) o `--read-only` (stdio) lo strumento non è offerto: utile per una **prova senza salvataggio**.
- Il server MCP manda al client anche istruzioni brevi sul flusso e sulle regole (non inventare valori, salvare solo se autorizzati, rileggere dopo un 409).
- Sign distance has no MCP tool: an agent never sets a zero (section 4; the REST API only reads it).

### Installare solo le dipendenze MCP

Sul server, nella cartella del repository, lo fa `helpers/setup_mcp.sh` (aggiunge `mcp` 2.x e le sue dipendenze a `.venv`, niente GPU; con uv se c'è, altrimenti pip):

```sh
helpers/setup_mcp.sh               # solo l'SDK
helpers/setup_mcp.sh --restart     # + copia del database se va migrato e riavvio di speedlimits-web: /mcp attivo
helpers/setup_mcp.sh --claude      # + Claude Code in questa cartella (server nnslr-review, permessi, effort medium)
```

A mano è `uv pip install --python .venv/bin/python -e ".[mcp]"` (oppure `.venv/bin/python -m pip install -e ".[mcp]"`), poi un riavvio del sito. Tutto quello che lo script installa o scrive fuori da git è elencato in [GUIDE.md](GUIDE.md#local-install-for-mcp), "Local install for MCP". `helpers/update.sh` non installa da solo l'extra `mcp`, ma non lo toglie. `GET /api/v1` dice `"mcp": true` quando è attivo. Per ritagli e ingrandimenti serve Pillow, già presente con l'extra `preannotate`.

### Claude Code: Streamable HTTP (dal PC, verso il servizio)

```sh
export NNSLR_REVIEW_TOKEN=...      # lo stesso del sito
claude mcp add --transport http nnslr-review http://<IP del server>:8765/mcp \
  --header "Authorization: Bearer $NNSLR_REVIEW_TOKEN"
claude mcp list                    # nnslr-review: ... (HTTP) - ✔ Connected
```

`claude mcp add` scrive il token in chiaro nella configurazione locale di Claude Code (`~/.claude.json`, non versionata). Per non scriverlo da nessuna parte usare un `.mcp.json` (nel proprio progetto, non in questo repository) con l'espansione delle variabili d'ambiente di Claude Code:

```json
{
  "mcpServers": {
    "nnslr-review": {
      "type": "http",
      "url": "http://<IP del server>:8765/mcp",
      "headers": {"Authorization": "Bearer ${NNSLR_REVIEW_TOKEN}"}
    }
  }
}
```

e avviare `claude` con `NNSLR_REVIEW_TOKEN` nell'ambiente.

### Claude Code: stdio (sul server, o dal PC via ssh)

```sh
# sul server stesso
claude mcp add nnslr-review -- ~/sunnypilot.speedlimits/helpers/mcp.sh
# dal PC: Claude Code avvia il server MCP sul server attraverso ssh (chiave ssh senza prompt)
claude mcp add nnslr-review -- ssh -T <utente>@<server> sunnypilot.speedlimits/helpers/mcp.sh
# prova senza salvataggio
claude mcp add nnslr-review-prova -- ssh -T <utente>@<server> sunnypilot.speedlimits/helpers/mcp.sh --read-only
```

Con stdio non serve token: chi può lanciare il comando sul server ha già accesso ai file. Il processo usa lo stesso database del servizio (SQLite in WAL, scritture serializzate e controllate con `base_revision`).

### Usarlo

In Claude Code gli strumenti si chiamano `mcp__nnslr-review__<nome>`. Claude Code chiede il permesso a ogni strumento la prima volta: è il punto in cui Cristian autorizza o no `save_frame_review`. Un esempio di richiesta:

> Usa nnslr-review. Route 0000011e--0c8825628a, prova senza salvataggio: per i prossimi 5 fotogrammi da rivedere guarda l'immagine, chiedi i vicini se il cartello è lontano o ambiguo e dimmi cosa confermeresti.

### Altri client MCP

- Qualsiasi client che supporta **Streamable HTTP** con header personalizzati: URL `http://<IP del server>:8765/mcp`, header `Authorization: Bearer <token>`. Il server è senza sessione (`stateless`) e risponde in JSON; `GET /mcp` (stream dal server) non serve e risponde 405.
- Qualsiasi client **stdio**: comando `ssh -T <utente>@<server> sunnypilot.speedlimits/helpers/mcp.sh` (oppure il percorso locale di `helpers/mcp.sh`).
- **Codex** (`~/.codex/config.toml`), via stdio:

  ```toml
  [mcp_servers.nnslr-review]
  command = "ssh"
  args = ["-T", "<utente>@<server>", "sunnypilot.speedlimits/helpers/mcp.sh"]
  ```

  Le versioni di Codex che supportano Streamable HTTP accettano `url = "http://<IP del server>:8765/mcp"` con il token letto da una variabile d'ambiente (`bearer_token_env_var = "NNSLR_REVIEW_TOKEN"`): verificare il nome dell'opzione nella documentazione della versione installata.
- **MCP Inspector**: `npx @modelcontextprotocol/inspector`, trasporto "Streamable HTTP", URL `/mcp`, header `Authorization`.

## 6. Avvio e prove in locale, senza GPU

Niente di questo usa la GPU, importa nel dataset o tocca il comma.

```sh
cd ~/sunnypilot.speedlimits
# test automatici (dati sintetici in cartelle temporanee); i test con l'SDK MCP si saltano se manca
.venv/bin/python -m pytest tests/review -q

# un sito di prova su un'altra porta e un data root di prova, solo su 127.0.0.1
export NNSLR_REVIEW_TOKEN=prova-locale-0123456789
.venv/bin/nnslr review --data-root /tmp/nnslr-prova --port 8799
curl -sS -H "Authorization: Bearer $NNSLR_REVIEW_TOKEN" http://127.0.0.1:8799/api/v1

# il server MCP stdio sullo stesso data root (scrive solo su stderr fuori dal protocollo)
.venv/bin/nnslr mcp --data-root /tmp/nnslr-prova --read-only
```

Un data root di prova si costruisce con le funzioni di `tests/review/review_helpers.py` (`make_report`). Non confermare fotogrammi reali per provare: usare `--read-only` o un data root di prova.

## 7. Errori comuni

| Sintomo | Causa e rimedio |
| --- | --- |
| 401 `unauthorized` | Token mancante o sbagliato, o mandato come cookie: usare `Authorization: Bearer $NNSLR_REVIEW_TOKEN`. |
| 400 `token_in_url` | Il token era nell'indirizzo: spostarlo nell'header. |
| 403 `cross_origin` | Richiesta da una pagina web di un'altra origine, o `Origin` impostato dal client: toglierlo. |
| 403 `host_not_allowed` | Sito solo locale (127.0.0.1) chiamato con un altro nome host: usare `127.0.0.1` o un tunnel ssh. |
| 404 `unknown_route` / `unknown_frame` | Route senza report o chiave inesistente: rileggere `GET /routes` o `list_frames`. |
| 409 `stale_report` | Il report della route è stato rigenerato: rileggere il fotogramma (nuovo `run_id`) e rifare la revisione. |
| 409 `revision_conflict` | Qualcuno ha salvato quel fotogramma nel frattempo: rileggerlo e decidere; non sovrascrivere senza autorizzazione. |
| 409 `human_review_exists` | Un'AI prova a cambiare una conferma umana: non farlo, salvo richiesta esplicita (`override_human_review`). |
| 409 `image_changed` | Il PNG originale non corrisponde al report: rigenerare il report. |
| 428 `base_revision_required` | Manca `base_revision`: rimandare quella letta (0 se mai salvato). |
| 400 `invalid_reviewer_kind` | Manca `reviewer_kind` (`human`/`ai`) in un salvataggio REST. |
| 400 `illegal_speed`, `unknown_enum`, `invalid_bbox`, `value_family_mismatch` | Valore non ammesso: vedere `GET /api/v1/schema`. |
| 501 `mcp_unavailable` | SDK MCP non installato: sezione 5, poi riavviare il sito. |
| 501 `image_processing_unavailable` | `crop`/`long_side` senza Pillow sul server: chiedere l'immagine intera o installare Pillow. |
| `nnslr mcp: the MCP SDK is not installed` | Come sopra, per stdio. |
| Claude Code: `Failed to connect` (HTTP) | Controllare `curl -H "Authorization: Bearer ..." http://<server>:8765/api/v1` e che dica `"mcp": true`. |
| Claude Code: `Failed to connect` (stdio via ssh) | `ssh -T <utente>@<server> sunnypilot.speedlimits/helpers/mcp.sh --help` deve funzionare senza password; niente nel profilo della shell deve scrivere su stdout. |
| Il database non si apre con un `nnslr` vecchio (`schema version 2`) | Il database è già migrato: aggiornare il codice (`helpers/update.sh`). |
| 403 `session_required` | Un indirizzo del revisore AI chiamato con il solo token: si usa dal sito dopo l'accesso, o dal sito su 127.0.0.1 (sezione 8). |
| 409 `no_ai_reviewer` | Nessun revisore AI scelto: Settings → AI reviewer, oppure `nnslr llm-provider use ID`. |
| 409 `ai_not_tested` | Il provider non ha passato il suo ultimo **Test** (o non è mai stato provato, o ha cambiato indirizzo o modello): non può diventare il revisore, né rispondere ad **Ask AI**, né salvare con `save: "yes"`. Rifare **Test** in Settings (`nnslr llm-provider test ID`). |
| 429 `ai_busy` | Due richieste all'AI sono già in corso: riprovare tra poco. |
| 502 `ai_unreachable` / `ai_http_error` / `ai_bad_answer` / `ai_redirect`, 504 `ai_timeout` | Il server del provider non risponde, rifiuta (`HTTP <codice>` e il suo messaggio), risponde senza JSON, rimanda a un altro indirizzo (mai seguito) o non risponde in tempo: controllare con **Test** in Settings. |

## 8. Il revisore AI del sito (Settings)

Il sito può far revisionare i cartelli a un'AI che vede le immagini, attraverso un server compatibile con l'API OpenAI (`/v1/models`, `/v1/chat/completions`) scelto in **Account → Settings → AI reviewer**: strata, un `llama-server` di llama.cpp con un modello visivo e il suo `--mmproj`, o un altro servizio. Il server del sito chiama quel provider; l'uso quotidiano è in [GUIDE.md](GUIDE.md), sezione 3, i comandi `nnslr llm-provider`, `nnslr ai-review` e `nnslr measure-ai-reviewer` in [COMMANDS.md](COMMANDS.md).

Questi indirizzi sono del **sito**, non di `/api/v1` né di MCP:

- cambiare i provider, provarli, elencarne i modelli, chiedere un suggerimento e avviare `ai_review` o `measure_ai_reviewer` richiede un **browser che ha fatto l'accesso**, o il sito su 127.0.0.1; con il solo token (`X-Review-Token`) la risposta è 403 `session_required`;
- con il token si possono soltanto **leggere** `GET /api/settings/ai`, `GET /api/ai/state`, `GET /api/route/<route>/ai-trial` e `GET /api/panel/ai-reviewers`, che non contengono segreti;
- ogni POST vuole la stessa origine (403 `cross_origin`) e JSON (415);
- le chiavi stanno solo in `data/web/private/llm-providers.json` (permessi 600, cartella 700) o nel file indicato da `NNSLR_LLM_PROVIDERS`: nessuna risposta, log del sito, log del pannello o resoconto le contiene, e il browser vede solo `has_key`. Se il provider ripete la chiave in un suo messaggio d'errore, il sito la mostra come `***`.

| Metodo e indirizzo | Chi | Corpo | Risposta |
| --- | --- | --- | --- |
| GET `/api/settings/ai` | sito o token | — | `{providers, reviewer, can_edit}` |
| POST `/api/settings/ai/providers` | accesso o 127.0.0.1 | `{id?, name, base_url, model, key?}` | `{ok, provider, providers, reviewer, can_edit}` |
| POST `/api/settings/ai/providers/<id>/delete` | accesso o 127.0.0.1 | `{}` | `{ok, providers, reviewer, can_edit}` |
| POST `/api/settings/ai/reviewer` | accesso o 127.0.0.1 | `{id}` (o `null`) | `{ok, providers, reviewer, can_edit}` |
| POST `/api/settings/ai/models` | accesso o 127.0.0.1 | `{id?, base_url?, key?}` | `{ok, models: [{id, sees_images}]}` |
| POST `/api/settings/ai/test` | accesso o 127.0.0.1 | `{id}` | `{ok, result, provider}`, 200 anche quando `result.ok` è falso |
| GET `/api/ai/state` | sito o token | — | `{available, can_ask, reason, provider}` |
| POST `/api/route/<route>/frame/<s>/<i>/ai-suggest` | accesso o 127.0.0.1 | `{run_id, detections?}` | il suggerimento (sotto) |
| GET `/api/route/<route>/ai-trial` | sito o token | — | `{ok, trial}`: l'ultima prova senza salvare sul report attuale, o `null` |
| GET `/api/panel/ai-reviewers` | sito o token | — | `{providers, reviewer, reports}` |
| POST `/api/panel/run/ai_review` | accesso o 127.0.0.1 | `{route, provider?, save?, limit?}` | `{ok, id, action, label}` |
| POST `/api/panel/run/measure_ai_reviewer` | accesso o 127.0.0.1 | `{provider?, limit?}` | `{ok, id, action, label}` |

- Un provider è `{id, name, base_url, model, has_key, is_reviewer, reviewer_name, updated_at, last_test}`. In `POST /providers` la `key` assente tiene quella salvata, `""` la toglie, un altro testo la imposta; `base_url` è http o https e finisce in `/v1`; l'`id` nasce dal nome e non cambia più.
- `models` manda la chiave salvata di `id` solo all'indirizzo salvato: a un indirizzo appena scritto non manda mai una chiave salvata.
- `test` manda al modello un fotogramma rivisto da una persona, senza le proposte, e dice se risponde in JSON e se legge il suo cartello di velocità (`sees_images`); se il server dichiara il modello solo testo, `sees_images` è `false` e il modello non viene interpellato.
- **Solo un provider che ha passato il suo ultimo Test** (`last_test.ok` vero) diventa il revisore, risponde ad **Ask AI** e salva con `save: "yes"`: gli altri ricevono 409 `ai_not_tested`. Un nuovo indirizzo o modello cancella il Test. Se il revisore scelto non lo passa più, `GET /api/ai/state` dà `available: false` con il motivo e il provider.
- **Il suggerimento non salva niente.** Restituisce i riquadri in pixel del fotogramma intero (`bbox_xyxy`), ognuno con `ai_action` `keep`, `correct`, `add` o `unjudged` (quelli tenuti o corretti mantengono il loro `detection_id`), più `deleted`, `no_sign`, `complete`, `warnings`, `comment`, `reviewer`, `seconds` e `message`. La pagina lo applica come modifiche non salvate, e il **Confirm** della persona lo salva come revisione umana. Il modello riceve l'immagine, i riquadri numerati e la seconda lettura come suggerimenti, con coordinate da 0 a 1000, mai i fotogrammi vicini. Al massimo due suggerimenti (o Test) alla volta: il terzo riceve 429 `ai_busy`. Un report rigenerato dà 409 `stale_report`.
- Le opzioni delle azioni sono stringhe (un numero o un booleano dà 400 `bad_option`): `route` una route con un report, `provider` un id di Settings (di default il revisore), `save` `yes` o `no` (di default `no`), `limit` da 1 a 5000 fotogrammi. Il lavoro esegue `helpers/ai_review.sh ROUTE --provider=ID [--save] [--limit=N]` o `helpers/measure_ai_reviewer.sh --provider=ID [--limit=N]`: la chiave la legge il comando dal file, mai dagli argomenti o dall'ambiente.
- **Le revisioni salvate da `ai_review` con `save: "yes"` sono definitive**: `reviewer` `ai/<provider>/<modello>`, `reviewer_kind` `ai`, canale `web`, `client` `ai-review`; il prossimo import le mette nel dataset come revisioni AI (`label_basis` `ai_review`). I fotogrammi confermati da una persona non vengono toccati. Senza `save` i suggerimenti vanno solo in `data/ai_reviews/<route>/`, e la pagina di revisione li mostra: `GET /api/route/<route>/ai-trial` dà l'ultima prova sul report attuale, `{route_id, run_id, provider: {id, name, model}, reviewer, started_at, finished_at, frames}`, con un suggerimento (come sopra, senza `ok`) per ogni fotogramma che nessuno ha ancora confermato.

## 9. Models › Drive: a recorded drive replayed (site addresses)

What it is for: before a set of networks goes on the car, the site replays a recorded drive through the car's own vision loop and shows the comma's screen over its video (the speed to hold, each sign as it is read). A drive where all of it is right is certified, and the next sets are then compared with it frame by frame: the regression test. The daily use is in [GUIDE.md](GUIDE.md), section 2 (**Drive**); the helper `helpers/simulate_drive.sh` and the commands `nnslr simulate` and `nnslr simulate-set` are in [COMMANDS.md](COMMANDS.md).

These addresses are the **site's**, not of `/api/v1` nor of MCP:

| Method and address | Who | Body | Answer |
| --- | --- | --- | --- |
| GET `/api/simulation/drives` | site or token | — | `{ok, sets, drives}` |
| GET `/api/simulation/drives/<route>?set=car\|in-use\|certified` | site or token | — | `{ok, route, sets, summary, view, differences}` |
| POST `/api/simulation/drives/<route>/certify` | sign-in or 127.0.0.1 | `{certified: true, set: "car"\|"in-use", made_at}`, or `{certified: false}` | `{ok, summary}` |
| POST `/api/panel/run/simulate` | site or token | `{set, route}`, or `{set, certified: "1"}` | `{ok, id, action, label}` |

- `sets` holds the two sets of networks: `car`, the release in the sunnypilot clone (what the comma downloads), and `in-use`, the versions in use on Models exported for the replays. Each is `{label, name, networks: [{role, file, id, sha256}], settings, ready, reason}` (`settings`: the car's settings in the release's `speed_limits.json`); `in-use` also has `wanted`, the versions in use by the car's names of the networks (`finder`, `classifier`, `decider`, `brain`), and is `ready` only while its export is of those versions and of the car's settings of today (a replay exports them again first).
- `drives` has every drive as `GET /api/speed-review/drives` lists it (`route`, `title`, `when`, `day`, `minutes`, `dashcam` and the rest), each with `sets` (its replays by set, `{made_at, name, frames, current}`; `current` is false for a replay made with other networks, another code of the car or other settings than today's), `certified` (`{at, who, set, name, networks, made_at, frames}`, or null) and `regression` (for each replay of a certified drive, `{frames, only_was, only_now, limit_frames, reading_frames, same_input, other_input, inputs_frames, sources, identical}`).
- For one drive, `summary` is its `{route, sets, certified, regression}` and `view` its replay with the set asked (`in-use` when none is said; `certified` is the certified replay), or null when the drive has none with it: `{route, set, label, name, made_at, networks, frame: [1344, 760], frames, series, events, signs}`. `series` are step series `[[seconds, value], ...]`, in seconds from the start of the drive's logs, named as the timeline's: `vis_limit` (the speed to hold in km/h, 0 for none), `vis_ahead`, `vis_ahead_dist`, `vis_road`, `vis_state`, `vis_reason`, `vis_raw_class`, `vis_raw_speed`, `vis_raw_value`, `vis_raw_unit`, `vis_raw_conf`, `vis_raw_taken`, `vis_raw_seen`, `vis_raw_road`, `vis_sign_dist`, `vis_sign_size`; a null value ends a series where the frames stop. `events` has `[seconds, "sign", kph]` for each reading. `signs` has `[seconds, [[x1, y1, x2, y2, class, confidence, road], ...]]`, SignNet's boxes of each frame read with any, in pixels of `frame`.
- `differences` is there for `set=car` or `set=in-use` on a certified drive that has that replay, else null: `{frames, only_was, only_now, limit: [[from_s, to_s, certified kph, now kph]], limit_frames, limit_seconds, reading: [[from_s, to_s, certified text, now text]], reading_frames, reading_seconds, same_input, other_input, inputs_frames, sources}`. A reading text is what the reading square says while a sign is in view (the sign, `other road`), with `read` on the frame it pulses, and `""` with no sign in view (the grey tail of the last sign is not compared again); its confidence and distance are left out, as they move by a point or a metre with any other network. `same_input` and `other_input` count the frames with another speed to hold where the brain was given the same rows, on the input groups both brains take, and the same picture since its memory last started again in both (the brain changed) or not (what is read before it changed, another video, or another `vision.py`); a step in one replay alone counts as another input; with the same brain in both, all are `other_input`. `inputs_frames` counts the frames where the car's signals kept with the replay differ (speed, steering, blinkers, yaw, pedals, openpilot on, the car ahead, GPS time, country); `sources` names the minutes whose video or log (by SHA-256) is not the one certified (`video of minute 3`, `log of minute 5`), and then the replay is never `identical`. A replay made before 10 Nov 2026 (format 1) is not read: the drive asks for **Compute** again.
- Errors of the reads: 400 `bad_set`; 409 `no_logs` when the drive's logs are not read yet (**Read the logs**, Speed review), since a replay is drawn on their clock.
- Certifying wants a **browser that has signed in**, or the site on 127.0.0.1: with the token alone the answer is 403 `session_required`. It also answers 400 `bad_request` (a body other than the two above), 409 `no_simulation` (the drive has no replay with that set), 409 `changed` (`made_at` is not the replay on the server: it was replayed again since the page showed it), 403 `actions_disabled` on a read-only panel. The certified file is the replay of that `made_at`, written whole from the same read. Every POST wants the same origin (403 `cross_origin`) and JSON (415).
- The options of the action are strings (a number or a boolean gives 400 `bad_option`, and nothing starts): `set` is `car` or `in-use` and is always said; then exactly one of `route` (a drive of this server) and `certified` (`"1"`: every certified drive; with none certified, 409 *Certify a drive first.*). The job runs `helpers/simulate_drive.sh --set=<set> --route=<route>` or `--set=<set> --certified`, on the V100 with the most free memory, shared with what runs there: when no card has about 2 GB free the answer is 409 `gpu_taken` with the room left, and a replay never runs on the CPU. One replay runs at a time. `GET /api/panel/status` shows it among `jobs` with its progress, `POST /api/panel/job/<id>/stop` stops it. It is a GPU job: an agent starts it only when asked to.
- A reviewer from outside gets none of these addresses (403).
