# Istruzioni per gli agenti

## ⚠️ VISION AND MAP ARE TWO SEPARATE SOURCES

**THE COMMA'S VISION LIMIT (`speedvisiond`, `vision.SpeedLimitMemory`) NEVER READS THE MAP. IT USES ONLY THE CAMERA AND
THE CAR'S OWN SIGNALS (SPEED, GYRO, STEERING ANGLE, BLINKERS). MAP AND VISION ARE WEIGHED AGAINST EACH OTHER ONLY BY
SUNNYPILOT'S RESOLVER ("MAP DATA + VISION"). NEVER PROPOSE A MAP FIELD (ITS LIMIT, `roadName`, ITS JUNCTIONS...) AS AN
INPUT OR A RULE OF THE VISION SIDE.** OpenStreetMap may only make training labels offline (road types, junctions).

- **Progress notes: [docs/progress/](docs/progress/).** Before Cristian turns the servers off, write there where the work stopped, in a note named `<YYYY-MM-DD>_<HH-MM>_CET.md`. The note says what is done, what was running (with the commands to finish it) and what is left; commit it and push it. `/tmp` is emptied when the servers go off, so copy anything worth keeping to `/root/work/`. At the start of a session, read the newest note and finish what it lists as running.
- Sul Mac di Cristian, prima di agire, leggere e applicare anche `/Users/cristianku/Gitlab/codex-contexts/AGENTS.md`; se lì non è leggibile, segnalarlo senza inventarne il contenuto. Sul server (il container dove girano i siti) quel file non esiste: non cercarlo e non segnalarlo.
- Per revisionare i cartelli, dalla pagina web o con MCP/REST API, leggere [review_instructions.md](review_instructions.md): descrive accesso con Playwright, MCP e REST (sezione 10), controllo delle immagini, annotazioni, salvataggio e verifica indipendente sul server.
- Un agente che lavora sul server stesso (questo clone, dove gira `speedlimits-web`) non chiede l'indirizzo né il token: la REST API è `http://127.0.0.1:8765/api/v1` e il token si carica dal `.env` del repository (`set -a; . ./.env; set +a`, poi l'header `Authorization: Bearer $NNSLR_REVIEW_TOKEN`), senza mai stamparlo. A Cristian chiede solo la route e se è una prova o una revisione con salvataggio, proponendo le route di `GET /api/v1/routes` con quanti fotogrammi restano da rivedere.
- Distinguere una prova del programma da una revisione reale. Salvare conferme nel database soltanto quando Cristian ha autorizzato la revisione con salvataggio. Non confermare immagini non esaminate.
- Una conferma salvata via REST API o MCP è definitiva: non esiste una seconda approvazione umana, e al prossimo import entra nel dataset di training come revisione AI (`ai/<nome>`, `label_basis` `ai_review`). Salvare non importa e non allena: import e training restano operazioni separate, che l'agente non lancia senza richiesta esplicita.
- Lavorare su `main`, preservando le modifiche esistenti. Non creare branch o worktree senza richiesta esplicita. Commit e push richiedono autorizzazione.
- La revisione autorizza le annotazioni richieste, non modifiche al codice del servizio, installazioni, riavvii, importazioni nel dataset, addestramenti o deployment.
- Gli aggiornamenti e le installazioni sul comma li esegue Cristian: non modificare o riavviare il dispositivo senza richiesta esplicita per quella specifica operazione.
- Non accendere mai `serverone-ai`, nemmeno indirettamente. Non usare Bonsai o worker RX6800; non contattare il worker V100 finché Cristian non ne comunica nuovamente la disponibilità.
- Non inserire token, cookie o credenziali nei file versionati, nei resoconti o nei log condivisi.
- Mai GitHub Actions: build, test e pubblicazione solo sul server. Non creare, ripristinare o riattivare niente in `.github/workflows/` e non trattare un push come qualcosa che avvia una build.

## Helper scripts e punti di ingresso

- Prima di usare un comando, consultare [README — Helpers, commands and scripts](README.md#helpers-commands-and-scripts), [docs/INSTALL.md](docs/INSTALL.md) e, per dataset/training, [docs/upstream-training-guide.md](docs/upstream-training-guide.md). Verificare gli argomenti con `nnslr <subcommand> --help` quando pertinente.
- La procedura d'uso quotidiana (helper, pannello, servizio `speedlimits-web`, token, aggiornamenti) è in [docs/GUIDA.md](docs/GUIDA.md). Ogni operazione ricorrente deve essere uno script del repository documentato lì: non lasciare passaggi fatti a mano senza traccia.
- Gli helper semplici `helpers/1_prepare.sh` (scrive nel data root, usa molta CPU), `helpers/2_review.sh` (apre la porta 8765 sulla LAN, crea il token in `.env`), `helpers/3_train.sh` (importa nel dataset, installa `.venv-gpu`, usa la GPU) e `helpers/update.sh` (`git pull`, aggiorna `.venv`, riavvia il servizio) hanno effetti reali come gli script sotto: non lanciarli senza richiesta. Il token in `.env` non va mai copiato in file versionati, resoconti o log. `helpers/status.sh` è solo lettura. Lo stesso vale per i pulsanti della pagina `/pannello` del sito di revisione: una revisione autorizzata non autorizza a premerli.
- `nnslr` è la CLI di questo progetto. Gli script `scripts/setup_v100_training_env.sh`, `scripts/run_v100_baseline.sh`, `scripts/publish_model.sh` e `scripts/install_review_service.sh` hanno effetti diversi (installazione dipendenze, consumo GPU, upload pubblico, modifica systemd): non eseguirli solo perché si sta consultando la documentazione. Preferire `--dry-run`/`--no-upload` quando disponibili e appropriati.
- Non confondere i progetti: `openpilot.distill` contiene gli helper video del modello di guida; la skill `OP-update-distilled` in `cristianku/openpilot_scripts` aggiorna le branch Distill di Sunnypilot/openpilot e **non** distribuisce i modelli di speed limit di questo repository.
- Per ogni affermazione su una feature installabile su comma, distinguere la ricerca/offline dal runtime effettivo; verificare prima che esistano backend, integrazione Speed Limit Assist e branch installabile. Non annunciare come funzionante lo snapshot advisory recuperato.
