# Istruzioni per l'agente che revisiona i cartelli

Queste istruzioni servono a un agente che apre il sito di revisione, guarda i fotogrammi delle route e conferma i cartelli di velocità. Valgono sia per un agente che lavora dentro Chrome con l'estensione di Claude o di ChatGPT, sia per un agente che pilota il browser con Playwright. Descrivono il sito nella versione del **23 ottobre 2026**.

L'indirizzo del sito lo indica Cristian, oppure il sito è già aperto in Chrome. Non cercarlo da solo e non scriverlo nei resoconti. Eccezione: un agente che gira sul server stesso usa `http://127.0.0.1:8765` (sezione 10).

Un agente che ha a disposizione il **server MCP** della revisione (strumenti `list_routes`, `next_frame`, `get_frame_image`, `save_frame_review`, ...) o la **REST API** `/api/v1` non ha bisogno del browser: segue la sezione 10, oltre alle regole delle sezioni 1, 4, 5 e 8. Le regole per riconoscere i cartelli (sezioni 4–5) sono le stesse.

## 1. Regole prima di iniziare

1. Se si lavora dal repository, leggere anche `AGENTS.md`. Un agente che ha solo il browser segue queste istruzioni e, nel dubbio, chiede a Cristian.
2. Capire quale delle due attività è stata chiesta:
   - **prova senza salvataggio**: si guarda e si modifica, ma non si preme **Conferma**. Il browser ricorda le modifiche anche dopo una ricarica. Alla fine premere **Annulla modifiche** su ogni fotogramma toccato, finché non restano tacche arancioni;
   - **revisione con salvataggio**: si conferma. Solo se Cristian l'ha chiesto esplicitamente per quella route. Ogni conferma salvata è definitiva, anche quella di un agente AI: nessuno la riapprova e al prossimo import diventa un dato di training.
3. Controllare il campo **Revisore** (in alto a destra). La pagina lo riempie da sola con l'ultimo nome usato in quel browser, quindi può contenere già `cristianku` o il nome di un'altra persona. Cancellarlo e scrivere un nome riconoscibile dell'agente, per esempio `Claude` o `ChatGPT`. Mai `cristianku` o altri nomi di persone. Ricontrollarlo dopo ogni ricarica. La pagina registra ogni conferma come revisione umana: per questo un agente AI, quando può, usa MCP o la REST API (sezione 10), che la registrano come revisione AI (`ai/<nome>`).
4. Non modificare né ritirare conferme già fatte da altri revisori, salvo richiesta esplicita che le comprenda.
   - Chi ha confermato un fotogramma si legge nella riga di stato in fondo al pannello: "Salvato ✓ · revisione N · <revisore>".
   - La pagina non si aggiorna da sola: le conferme fatte da altri dopo l'apertura compaiono solo ricaricando. Se altri possono lavorare sulla stessa route, ricaricare prima di iniziare e ogni tanto durante il lavoro.
5. Nel **Pannello** non premere nessun pulsante (Prepara, Importa, Allena, Aggiorna il codice): hanno effetti reali sul server. Non riavviare il server, non toccare il comma, non importare nel dataset, non avviare training.
6. Il nome, la password e il token di accesso non vanno mai copiati in file, resoconti, screenshot o log.
7. Non confermare mai un fotogramma senza averlo guardato davvero. Non accettare in blocco le proposte del modello.

## 2. Accesso

1. Aprire l'indirizzo ricevuto (`http://<IP del server>:8765/`). Il sito chiede **nome e password**: nel Chrome di Cristian di solito è già connesso. Un agente con Playwright non usa la password: manda il token di accesso nell'header `X-Review-Token` di ogni richiesta (`browser.newContext({extraHTTPHeaders: {'X-Review-Token': process.env.NNSLR_REVIEW_TOKEN}})`), con il token configurato da Cristian nell'ambiente, mai nell'indirizzo. Se compare la pagina di accesso o **Accesso richiesto**, chiedere a Cristian.
2. La pagina iniziale **Route** elenca le route con un report. Ogni card mostra "N di M confermati", i sospetti e un'azione: **Inizia →**, **Riprendi →** o **Ricontrolla →**. Aprire la route indicata da Cristian.
3. Lavorare con la **pagina** larga almeno 1000 px. Con il pannello laterale dell'estensione aperto la pagina è più stretta della finestra: allargare la finestra.

   Sotto i 981 px la pagina passa al layout da telefono: sotto la foto vengono subito i cartelli; timeline, legenda e "Confermati con dubbi" sono chiusi sotto il pulsante **Timeline** (lo stato aperto/chiuso resta nel browser). Il pulsante Conferma diventa la barra fissa in basso, tra ‹ e ›, e dice cosa salva in breve: **Conferma · 60**, **Nessun cartello**, **Avanti →** (cioè "Già confermato · avanti →") o **Salvataggio…**.
4. Se in fondo al pannello compare **"Revisioni non lette dal server"** o **"Il report di questa route è stato rigenerato: ricarica la pagina"**, fermarsi: ricaricare la pagina e, se resta, avvisare Cristian.
5. La pagina della route riapre con il filtro e il fotogramma usati l'ultima volta in quel browser. All'inizio scegliere il filtro **Da rivedere**.

## 3. Com'è fatta la pagina di revisione

**Barra in alto**
- Il menu della route.
- I filtri **Da rivedere / Sospetti / Con proposte / Tutti** e, solo quando ce ne sono, **AI · N**:
  - **Da rivedere**: i fotogrammi non ancora confermati;
  - **Sospetti**: modello e seconda lettura non sono d'accordo, oppure un fotogramma vicino ha un cartello (confermato o visto dalla seconda lettura) e questo non ha niente. Il motivo è nella riga viola **Sospetto: …**. Un fotogramma confermato non è mai sospetto, ma una conferma può rendere sospetti i vicini, quindi il numero dei sospetti non scende sempre;
  - **Con proposte**: il modello ha proposto almeno un riquadro;
  - **AI · N**: i fotogrammi confermati da un revisore AI (REST API o MCP). La conferma AI è definitiva come quella di una persona ed entra nel dataset al prossimo import, senza approvazione. La pagina ne mostra solo la provenienza: stato **Confermato · AI**, tacca a righe e riga di stato "Salvato ✓ · revisione N · ai/<nome> (AI · MCP)" (o "· REST"). Il pulsante dice **Già confermato · avanti →** come per gli altri confermati. Una persona può correggerla dalla pagina: la correzione diventa una revisione umana.
- Il contatore "N di M confermati · K sospetti".
- Il campo **Revisore**.

**A sinistra**
- **La foto**, con i riquadri:
  - etichette gialle come `1 · 40`, con il numero del riquadro e il valore;
  - riquadri azzurri tratteggiati con etichette come `60?`: sono la seconda lettura, solo un aiuto.
- **La barra sotto la foto**:
  - le frecce **‹ ›** con la posizione nel filtro, per esempio `62 / 68`;
  - lo stato del fotogramma: Da rivedere, Da confermare, Da riconfermare, Salvataggio…, Confermato (Confermato · AI se l'ha confermato un revisore AI), Non salvato;
  - "Segmento X · immagine Y · N proposte";
  - **+ Riquadro**, **Elimina** (elimina il riquadro selezionato) e l'interruttore **Seconda lettura**.
- **La timeline**: una riga di tacche per segmento ("seg 2", "seg 3"…), una tacca per immagine, in ordine.
  - Le tacche non hanno scritte. Passandoci sopra col mouse, o leggendo l'albero di accessibilità, ognuna dice per esempio "Segmento 2 · immagine 6 · confermato".
  - Cliccando una tacca si apre quel fotogramma: controllare poi "Segmento X · immagine Y" sotto la foto.
  - I colori:
    - verde: confermato;
    - verde con la parte alta gialla: confermato con un cartello senza numero o illeggibile;
    - verde a righe: confermato da un revisore AI (REST API o MCP);
    - viola: sospetto;
    - giallo: con proposte o seconda lettura;
    - arancione: modificato, da confermare;
    - rosso: non salvato;
    - grigio: niente proposto.
- **Confermati con dubbi:** tre contatori cliccabili. Contano fotogrammi, non riquadri:
  - **senza numero**: un Limite massimo, una Zona, un Segnale variabile o un Numero a terra con **?**. La Fine limitazione non conta;
  - **con illeggibili**: almeno un riquadro Illeggibile;
  - **strada non indicata**: un Limite massimo, una Zona o un Segnale variabile con strada **Da chiarire**. È il valore iniziale di ogni riquadro, quindi questo contatore sale se non si sceglie la strada. La strada non cambia il colore della tacca.

  Ciascuno filtra i fotogrammi confermati che hanno quel dubbio; un secondo clic torna a "Da rivedere".

**A destra, il pannello**
- **Cartelli**: un'eventuale riga viola **Sospetto: …** con il motivo, poi un riquadro per riga:
  - numero e descrizione, per esempio "Limite massimo · 40 km/h";
  - sotto, l'origine: "modello · 92%", "modello 40 → 60" (cioè hai cambiato il valore proposto) oppure "aggiunto a mano";
  - **"stesso cartello di N"** in arancione: è un doppione di un altro riquadro;
  - **✕**: elimina quel riquadro.
- **Riquadro N**, quando un riquadro è selezionato:
  - lo zoom sul cartello, con gli angoli gialli per ridimensionare;
  - i cartelli rapidi **10 20 30 40 50 60 70 80 100 120** e **?** (numero non leggibile);
  - in the US (the frame's country, picked or from its GPS) the quick signs are the white SPEED LIMIT rectangles in mph, **15 20 25 … 75 80** and **?**: the value is stored as written with `sign_unit` `mph` (25 mph is 25), and a speed takes two keys (**3 5** = 35; **3** alone = 30);
  - **Altro valore**, per i numeri che non ci sono tra quelli rapidi. Compare solo quando il tipo ammette un numero;
  - **Tipo**: Limite massimo, Fine limitazione, Illeggibile, Segnale variabile, Zona; più in piccolo Numero a terra, Altro cartello, Non è un cartello;
  - **Strada**: La nostra, Un'altra, Da chiarire;
  - **Panel below**, always open: None, Hours (a limit valid only at some hours, such as the German 22-6), Wet (a limit valid only on a wet road, such as the German "bei Nässe"), Other. With None and Other the box stays on the disc; with Hours and Wet it takes the disc and the plate, one box for both, and the page says so under the row. Under it **From** and **to** take the hours written on the plate, whole hours (22 and 6 for "22-6"): leave them empty when they cannot be read, or the plate has days, minutes or two ranges. Fit box (C, ⇧C) leaves such a box as it is;
  - **Dettagli del cartello**, chiuso: visibilità, occlusione, pannello sotto, temporaneo, display variabile, note.
- **Seconda lettura**: c'è solo sui fotogrammi che ne hanno una. Ogni voce indica tipo, valore, sicurezza, letture (A, B) e lato, con il pulsante **Usa**, che diventa **Già usato** quando quel cartello c'è già.
- **Altro**, chiuso: sito fisico, incontro, esportazione JSONL, scorciatoie.
- **In fondo**, sempre nello stesso punto:
  - la riga di stato;
  - il link **Annulla modifiche**, solo su un fotogramma modificato o non salvato;
  - il link **Annulla conferma**, solo su un fotogramma confermato sul server;
  - il pulsante **Conferma**, che dice cosa salva: "Conferma · 60 · 40", "Conferma · nessun cartello", oppure "Già confermato · avanti →" su un fotogramma già confermato e non modificato.

La foto e il pulsante Conferma non si spostano mai da soli: avvisi ed errori compaiono nel pannello, sopra la riga di stato.

## 4. Come guardare le immagini

- Per leggere un cartello selezionarne il riquadro e usare lo zoom nel pannello. La foto intera serve per il contesto: strada, corsie, rami laterali.
- Lo zoom e uno screenshot ridotto possono non bastare per un numero piccolo. Se resta un dubbio sostanziale, non inventare: usare **?** o **Illeggibile**.
- Le proposte del modello e la seconda lettura sono punti di partenza, non verità. Controllare sempre che il riquadro racchiuda il cartello giusto.
- Prima di confermare un fotogramma senza cartelli, controllare tutta la foto: bordi, cartelli lontani e piccoli, portali sopra la strada, numeri dipinti sull'asfalto.

## 5. Criteri di revisione

- Dare un valore solo quando il numero si legge in quel fotogramma. Non copiarlo dai fotogrammi vicini né dalla seconda lettura.
- Se si riconosce un limite ma non il numero: tipo **Limite massimo** e cartello **?**. Se non si capisce neppure che cartello è: **Illeggibile**, con una nota nei dettagli.
- Distinguere un limite attivo da una **Fine limitazione**. La fine limitazione si salva senza numero, come nella revisione del 21 ottobre 2026: scegliere **Fine limitazione**, poi premere **?** per togliere il numero, e scrivere l'eventuale numero barrato nelle note. Passando a Fine limitazione la pagina tiene il numero scelto prima: controllare che la riga dica solo "Fine limitazione".
- Distinguere **Zona** da un normale limite massimo.
- **Strada**:
  - **La nostra**: il limite vale per la strada che si sta percorrendo;
  - **Un'altra**: per esempio una zona 30 su un accesso laterale;
  - **Da chiarire**: quando non si può stabilire, per esempio un ramo oltre una rotonda.
- Un pannello variabile spento o vuoto non mostra una velocità: tipo **Segnale variabile**, cartello **?** e una nota. Un segnale luminoso sovraesposto resta illeggibile anche se la proposta suggerisce un numero.
- Non sono cartelli di velocità e vanno eliminati con **✕**: loghi, pubblicità, precedenze, frecce, limiti di altezza e altri divieti. **Altro cartello** e **Non è un cartello** si usano solo se Cristian lo chiede: tengono il riquadro come esempio.
- I doppioni sullo stesso cartello ("stesso cartello di N", per esempio 40 e 80 sullo stesso disco) si eliminano con **✕**. Si tiene un solo riquadro, con il valore giusto. "Stesso cartello di N" compare solo sul secondo riquadro della coppia e non dice quale dei due sia sbagliato: deciderlo guardando.
- Controllare anche i riquadri piccoli contenuti in uno più grande.
- Un fotogramma senza cartelli pertinenti si conferma comunque, con **Conferma · nessun cartello**: anche gli esempi negativi servono.

Esempi visti nelle route reali:
- una fine limite generale 50 proposta come 40;
- limiti 60 proposti come 40/80;
- loghi aziendali proposti come limiti;
- un limite 10 all'ingresso di un parcheggio, accanto a un divieto di altezza 2,3 m.

## 6. Procedura per ogni fotogramma

1. **Aprire il fotogramma**: con le frecce **‹ ›**, con una tacca della timeline o, per un agente che usa JavaScript, per chiave (sezione 9). Il filtro **Da rivedere** mostra solo ciò che resta da fare.
2. **Aspettare la foto**. I riquadri compaiono solo quando la foto nuova è caricata. Controllare che "Segmento X · immagine Y" sia il fotogramma voluto.
3. **Correggere i riquadri esistenti.** Cliccare la riga nella lista (o il riquadro sulla foto), poi:
   - scegliere il valore con un cartello rapido o con **?**;
   - per un numero che non c'è tra quelli rapidi:
     - scriverlo in **Altro valore** e premere **Invio**, prima di cliccare qualunque altro pulsante. Se si clicca subito altrove, quel primo clic serve solo a registrare il numero e va perso;
     - controllare che la riga del riquadro mostri il nuovo valore, per esempio "Limite massimo · 90 km/h";
     - se il numero non è un intero tra 5 e 250, sotto il campo compare in rosso "Scrivi un numero intero tra 5 e 250, oppure lascia vuoto se non si legge." e il riquadro **tiene il valore di prima**: Conferma salverebbe quello, senza nessun errore;
   - scegliere il **Tipo**. Il valore e il tipo si influenzano:
     - un cartello rapido, anche **?**, trasforma da solo in **Limite massimo** un riquadro Illeggibile, Altro cartello o Non è un cartello. Per **Zona** o **Segnale variabile** premere poi il tipo;
     - i tipi senza numero (Illeggibile, Altro cartello, Non è un cartello) cancellano il valore: se si torna indietro, il numero va scelto di nuovo;
   - scegliere la **Strada**. Ogni riquadro parte con **Da chiarire** già premuto, anche quello disegnato o preso con **Usa**. Per ogni cartello di velocità premere **La nostra** o **Un'altra**, e lasciare **Da chiarire** solo quando lo si è deciso davvero;
   - se il riquadro non racchiude bene il cartello, trascinare gli angoli gialli, meglio nello zoom, un angolo alla volta. Dopo ogni trascinamento lo zoom si ricentra sul riquadro nuovo: gli angoli tornano più o meno nello stesso punto ed è il cartello a sembrare più piccolo. Fare un nuovo screenshot, controllare i bordi e solo dopo trascinare l'angolo successivo;
   - eliminare con **✕** i falsi positivi e i doppioni, uno alla volta. Dopo ogni ✕ i riquadri si rinumerano (anche in "stesso cartello di N") e la ✕ della riga successiva finisce sotto il puntatore. Fare un nuovo screenshot prima di ogni ✕, oppure eliminare partendo dall'ultima riga.
4. **Aggiungere i cartelli mancanti**:
   - premere **+ Riquadro**. La foto mostra "Trascina sull'immagine per disegnare il riquadro" e il pulsante diventa **Annulla**: premerlo per uscire senza disegnare;
   - trascinare sulla foto da un angolo all'opposto del cartello;
   - il riquadro nuovo nasce come **Illeggibile**, con strada **Da chiarire**, e resta selezionato. Assegnargli valore, tipo e strada come al punto 3. Per lasciarlo Illeggibile non premere nessun cartello rapido: anche **?** lo trasformerebbe in Limite massimo senza numero.

   In alternativa, se la seconda lettura ha visto il cartello, premere **Usa** sulla sua voce. **Usa** aggiunge un riquadro nuovo ("aggiunto a mano", strada **Da chiarire**) e non corregge quello del modello. Se compare "stesso cartello di N", eliminare con **✕** il riquadro sbagliato. Poi controllare valore, tipo e strada.
5. **Confermare**:
   - verificare che il campo **Revisore** contenga il nome dell'agente;
   - premere **Conferma · …**.

   Il risultato:
   - **Riuscita**: in fondo compare in verde "Salvato `<segmento/immagine>` alle HH:MM. Ecco il prossimo." e la pagina passa da sola al fotogramma successivo del filtro. La tacca diventa verde (con la parte alta gialla se restano dubbi). Il contatore in alto sale solo per un fotogramma non ancora confermato: su una riconferma resta uguale.
   - **Fallita**: si resta sul fotogramma, lo stato diventa **Non salvato** e la riga di stato dice in rosso "NON salvato: motivo". Vedere la sezione 7.
   - **Già confermato · avanti →**: il fotogramma era già confermato e non è stato modificato. Non si salva niente, si passa al successivo.
6. **Annotarsi** l'esito: chiave del fotogramma, riquadri salvati e dubbi. Servono per il resoconto (sezione 8).

**Tastiera.** Non usarla per navigare o confermare: usare sempre i pulsanti. Scrivere solo dopo aver cliccato dentro un campo di testo. Fuori dai campi:
- **Invio** conferma il fotogramma, anche subito dopo un clic su una freccia o su + Riquadro;
- **Canc** e **Backspace** eliminano il riquadro selezionato;
- le frecce **← →** cambiano fotogramma;
- **N** avvia il disegno, **H** nasconde la seconda lettura, **Esc** toglie la selezione.

Dentro **Altro valore** e **Note**, invece, **Invio** registra il valore e non conferma.

## 7. Errori e casi particolari

- **"Scrivi nome o sigla del revisore, poi conferma."**: il campo Revisore è vuoto. Compilarlo e premere di nuovo **Conferma**.
- **"NON salvato: …"**: il motivo è scritto nella riga.
  - "Valore non ammesso in uno dei campi a scelta.": correggere il riquadro e riconfermare. Un numero fuori 5–250 non arriva fin qui: lo blocca già **Altro valore** (sezione 6).
  - "salvato da un'altra pagina nel frattempo": qualcun altro ha cambiato quel fotogramma. Ricaricare la pagina: la modifica locale viene scartata e si vede la versione del server. Non sovrascriverla senza autorizzazione.
  - "server non raggiungibile": aspettare e premere **Riprova a salvare** nel riquadro rosso. Se continua, fermarsi e avvisare Cristian.
- **Riquadro rosso "N fotogrammi confermati NON salvati sul server"** (con uno solo: "1 fotogramma confermato NON salvato sul server (chiave)."): ci sono conferme rimaste nel browser. Aprire quel fotogramma e leggere il motivo nella riga di stato.
  - **Riprova a salvare** serve solo quando il motivo è "server non raggiungibile" o un errore del server. La pagina riprova anche da sola ogni 30 secondi, quindi il riquadro può sparire senza clic. Quando riesce si resta sullo stesso fotogramma e la riga dice "Salvato ✓ · revisione N · <revisore>".
  - Con un valore non ammesso o con "salvato da un'altra pagina", riprovare dà lo stesso errore: correggere e riconfermare, oppure ricaricare, come sopra.

  Il lavoro è finito solo quando il riquadro rosso sparisce.
- **Annulla modifiche** riporta il fotogramma all'ultima versione confermata, o alle proposte se non era mai stato confermato.
- **Annulla conferma** va premuto due volte: il primo clic lo trasforma in **Sì, annulla**, il secondo toglie la conferma. Va usato solo se richiesto.
- **Modificare non è salvare.** Un fotogramma modificato e non confermato resta arancione "Da confermare". Prima di finire non ne devono restare.
- "Non restano fotogrammi da rivedere con questo filtro." vuol dire che non ce ne sono più **dopo** questo: la pagina non ricomincia dall'inizio. Controllare la posizione `N / M` sotto la foto e la timeline. Se ne restano prima, aprirli con una tacca o con **‹**.

## 8. Controllo finale e resoconto

1. Ricaricare la pagina. Controllare il contatore "N di M confermati" e che non ci siano riquadri rossi né tacche rosse o arancioni.
2. Aprire il filtro **Da rivedere**: deve restare vuoto, o contenere solo quello che si è deciso di lasciare.
3. Guardare i contatori **Confermati con dubbi** e ricontrollare almeno alcuni di quei fotogrammi.
4. Per gli agenti che possono fare richieste HTTP: `GET /api/route/<route_id>/reviews`, con la stessa sessione del browser, restituisce:
   - `confirmed_count`;
   - la lista `reviews` (`frame_key`, `confirmed`, `reviewer`, `reviewer_kind`, `channel`, `revision`, `detections`), comprese le conferme annullate (`confirmed: false`). `reviewer_kind` è `human` o `ai`, `channel` è `web` (pagina), `rest` o `mcp`.

   Farla anche all'inizio, prima di confermare, e conservare `frame_key`, `reviewer`, `revision` e `confirmed`. Alla fine le voci di altri revisori devono avere la stessa `revision`.
5. Riferire a Cristian:
   - route;
   - fotogrammi confermati da te, con e senza cartelli;
   - quanti riquadri per tipo;
   - quanti con numero sconosciuto, illeggibili e con strada da chiarire;
   - errori incontrati e come sono stati risolti;
   - fotogrammi lasciati in sospeso e perché.

   Lo "0 sospetti" non dimostra che le etichette siano giuste: dirlo esplicitamente.
6. Il browser ricorda il nome del revisore. Se si è lavorato nel Chrome di Cristian, alla fine svuotare il campo **Revisore**: altrimenti le sue prossime conferme risultano firmate dall'agente.

## 9. Per agenti che usano JavaScript o Playwright

La pagina espone il proprio stato con attributi che non cambiano il layout. Conviene usarli al posto di colori, testi e coordinate:

| Dove | Attributo | Significato |
| --- | --- | --- |
| `body` | `data-frame-key` | Fotogramma mostrato (`segmento/immagine`) |
| `body` | `data-frame-status` | `todo`, `dirty`, `saving`, `error`, `done` |
| `body` | `data-frame-reviewer-kind` | Chi ha la conferma sul server: `human`, `ai` (revisore AI, conferma definitiva come `human`), vuoto se non confermato |
| `body` | `data-server` | `loading` finché le revisioni non arrivano dal server, poi `ready`; `error` o `stale` (report rigenerato); `offline` per la pagina aperta senza server |
| `body` | `data-unsaved` | Fotogrammi confermati non ancora salvati (`0` = tutto salvato) |
| `body` | `data-last-save-key`, `data-last-save-result`, `data-last-save-revision`, `data-save-count` | Ultimo salvataggio. Il risultato vale `saving`, poi `ok` o `error`; `unchanged` se la conferma non ha inviato niente; `withdrawn` dopo un ritiro; `need-reviewer` se manca il revisore |
| `#stage` | `data-photo-key` | Chiave del fotogramma la cui foto è caricata (`error:<chiave>` se la foto non si carica) |
| `#boxList .box-row` | `data-index`, `data-detection-id`, `data-duplicate-of` | Riga di un riquadro. `.box-main` lo seleziona, `.box-del` lo elimina. `data-duplicate-of` è l'indice (da 0) dell'altro riquadro, cioè N−1 di "stesso cartello di N", e c'è solo sulla seconda riga della coppia: non dice quale dei due è sbagliato |
| `#families`, `#familiesMore` | `[data-family]` | Tipo di cartello |
| `#quickValues` | `[data-value]` | Velocità rapide; `data-value="unknown"` = numero non leggibile |
| `#roads` | `[data-applicability]` | `own_road`, `other_road`, `unresolved` |
| `#hintList` | `[data-hint-index]` | Pulsante **Usa** di una voce della seconda lettura |
| `#timeline` | `button[data-key]` | Tacca di un fotogramma |
| `#confirm`, `#mConfirm` | `data-action="confirm"`, `data-save-state` | Usare `[data-action="confirm"]:visible`: sotto i 981 px è visibile solo quello della barra in basso |
| `#withdraw` | `data-armed` | Annulla conferma: `true` dopo il primo clic |

Note:
- Anche i riquadri disegnati sulla foto hanno `data-index`. Per la lista usare sempre `#boxList .box-row[...]`, meglio con `data-detection-id`, perché l'indice cambia dopo ogni eliminazione.
- I campi di **Dettagli del cartello** e di **Altro** sono dentro sezioni chiuse: aprirle (clic sul titolo) prima di usarli.
- Dopo `fill` su `#value` o `#notes` premere `Enter` (o lanciare l'evento `change`) prima di cliccare altro: il valore si registra solo così.

Schema per un fotogramma, con Playwright:

```js
const key = '3/28';
await page.locator('#timeline button[data-key="' + key + '"]').click();
await page.waitForSelector('body[data-frame-key="' + key + '"]');
await page.waitForSelector('#stage[data-photo-key="' + key + '"], #stage[data-photo-key="error:' + key + '"]');
// correzioni: per esempio il primo riquadro diventa 60, sulla nostra strada
await page.locator('#boxList .box-row').first().locator('.box-main').click();
await page.locator('#quickValues [data-value="60"]').click();
await page.locator('#roads [data-applicability="own_road"]').click();
// conferma e attesa del risultato di QUEL fotogramma
await page.locator('[data-action="confirm"]:visible').click();
await page.waitForSelector('body[data-last-save-key="' + key + '"]:not([data-last-save-result="saving"])');
const result = await page.evaluate(() => document.body.dataset.lastSaveResult);   // 'ok', 'error', 'unchanged', 'need-reviewer'
```

Più indicazioni:
- **Attese.** Dopo il caricamento della route aspettare `body[data-server="ready"]` prima di leggere conteggi o modificare. Prima del controllo finale aspettare `body[data-unsaved="0"]`.
- **Un riquadro nuovo.** Premere `#draw`, poi trascinare con il mouse sulla foto. Le coordinate dei riquadri (`bbox_xyxy = [x1, y1, x2, y2]`) sono quelle dell'immagine originale, non dello schermo. Per convertire un punto dell'immagine in coordinate dello schermo:

  ```js
  const [sx, sy] = await page.evaluate(([x, y]) => {
    const svg = document.getElementById('scene');
    const p = svg.createSVGPoint(); p.x = x; p.y = y;
    const q = p.matrixTransform(svg.getScreenCTM());
    return [q.x, q.y];
  }, [x, y]);
  ```

  La stessa conversione vale per lo zoom, con `document.getElementById('zoom')`. Per ridimensionare un riquadro selezionato premere il mouse esattamente su un suo angolo (per esempio `x2, y2` per quello in basso a destra), trascinarlo nel punto nuovo e poi controllare `bbox_xyxy`.
- **Navigazione.** Dopo ogni conferma riuscita la pagina passa da sola al fotogramma successivo. Aprire sempre il prossimo per chiave e controllare `body[data-frame-key]`, invece di premere ‹ › o Invio.
- **Funzioni interne.** Esistono anche funzioni come `openFrame(key)`, `rows`, `S`, `workOf(row)`, `addManualBox(row, box, family, value)` e `render()`, ma sono dettagli interni che possono cambiare. Se una modifica passa da `workOf(...)`, chiamare poi `render()`. Non modificare mai SQLite o i report sul server.
- **Valori ammessi dal server.** Se non sono rispettati, il server risponde con un errore 400:

  | Campo | Valori |
  | --- | --- |
  | `family` | `maximum_speed`, `cancellation`, `zone`, `variable_display`, `road_marking_candidate`, `other_sign`, `unreadable`, `not_a_sign` |
  | `value_kph` | intero tra 5 e 250, oppure `null`. Un numero è ammesso solo con `maximum_speed`, `cancellation`, `zone`, `variable_display`, `road_marking_candidate`; con `other_sign`, `unreadable`, `not_a_sign` deve essere `null` |
  | `applicability` | `own_road`, `other_road`, `unresolved` |
  | `visibility` | `unknown`, `clear`, `partial`, `poor` |
  | `occlusion` | `unknown`, `none`, `partial`, `heavy` |
  | `supplementary_panel` | `unknown`, `none`, `time` (hours, e.g. 22-6: one box around disc and plate), `wet` (only on a wet road, e.g. "bei Nässe": one box around disc and plate), `present` (another panel: the box stays on the disc), `unresolved` |
  | `plate_hours` | with family `maximum_speed` and panel `time`: the hours on the plate, `22-6` (from-to, whole hours 0-23, no leading zero); null when they cannot be read, or the plate has days, minutes or two ranges |
  | `temporary`, `variable_display` | `true`, `false` oppure `null` |
  | `notes` | testo, anche vuoto; mai `null` |
  | `bbox_xyxy` | dentro l'immagine, con `x1 < x2` e `y1 < y2` |

  Al massimo 100 riquadri per fotogramma; il nome del revisore al massimo 100 caratteri.
- **Stato iniziale.** Il JSON dei fotogrammi è nello script `#payload` della pagina ed è grande: non stamparlo tutto, estrarre solo chiavi, `preview_path` e numero di proposte. Le revisioni non sono lì: leggerle con `GET /api/route/<route_id>/reviews` (sezione 8).
- **Sessione.** Per una sessione lunga conservare lo stesso `context` (cookie) e registrare gli eventi `pageerror` e le risposte HTTP con stato ≥ 400, stampando solo stato e percorso, senza query string né token.

## 10. Per agenti che usano MCP o la REST API

Il server MCP (`/mcp` del sito, oppure `helpers/mcp.sh` via ssh) e la REST API `/api/v1` danno gli stessi dati della pagina senza browser. Configurazione, indirizzi ed errori sono in [docs/API_MCP.md](docs/API_MCP.md); qui c'è come lavorare.

**Prima di iniziare**
1. Capire, come nella sezione 1, se è una **prova senza salvataggio** o una **revisione con salvataggio** autorizzata da Cristian per quella route. In una prova non chiamare mai `save_frame_review` (REST: `POST .../review`); se disponibile, usare il server in sola lettura (`--read-only`), che non offre lo strumento.
2. Il token, se serve (HTTP), lo configura Cristian nel client: non chiederlo nella conversazione e non scriverlo in file, resoconti, log o indirizzi. Un agente che gira sul server stesso non chiede né indirizzo né token: usa `http://127.0.0.1:8765/api/v1` e carica il token dal `.env` del repository con `set -a; . ./.env; set +a`, senza stamparlo.
3. Se Cristian non ha indicato la route, elencare quelle con fotogrammi da rivedere (`GET /api/v1/routes`, MCP `list_routes`) e chiedergli quale, insieme a prova o salvataggio.
4. Le azioni del pannello non sono raggiungibili da MCP e API: non cercarle.
5. **Sign distance: agents never set zeros.** The zeros of Data › Sign distance (the frame where a sign is about to leave the picture) are marked only by a person signed in on the site. There is no MCP tool and no REST write for them, and an agent never sends the page's `POST /api/sign-distance/...` either, not even with a session; it never marks a zero, Can't tell, Clear or Undo in the site's data (only a UI test on a scratch test site, with a data root of its own, may). Reading is fine: `GET /api/v1/sign-distance` and `GET /api/v1/sign-distance/<route>` ([docs/API_MCP.md](docs/API_MCP.md), section 4).

6. **Speed review: agents never write.** The corrections of Data › Speed review (the blocks that give another speed than the driver's own from a time to a time of a drive, or leave it out of training, the roadworks marked on it, and how far it was checked for roadworks) are written only by a person signed in on the site. There is no MCP tool, no REST write and no command for them.

**Per ogni fotogramma**
1. `next_frame(route_id)` (REST: `GET /api/v1/routes/<route>/next`) dà il prossimo fotogramma non confermato, con le proposte del modello (`proposals`), la seconda lettura (`second_reading`), i motivi di sospetto (`suspicious_reasons`), l'eventuale revisione salvata, `run_id` e `revision`. Dopo un'interruzione basta richiamarlo: riparte da dove il lavoro si era fermato. Per andare avanti senza salvare (prova): `next_frame(route_id, after=<frame_key>)`.
2. **Guardare l'immagine** con `get_frame_image` (REST: `.../image`). Proposte e seconda lettura sono solo indizi. Per un numero piccolo chiedere un ritaglio ingrandito: `variant` `original`, `crop` = il riquadro allargato di circa metà per lato, `long_side` 512. Prima di confermare "nessun cartello" guardare tutta l'immagine (sezione 4).
3. Se il cartello è lontano, coperto, sfocato o ambiguo, `get_frame_context` (con `include_images` solo quando serve) mostra i fotogrammi vicini, anche nel segmento prima o dopo. I vicini aiutano a capire che cosa si sta guardando, ma il valore si scrive solo se si legge **in questo fotogramma** (sezione 5).
4. Decidere i riquadri con i criteri della sezione 5, con i nomi della tabella "Valori ammessi" della sezione 9 (`get_review_schema` li elenca):
   - tenere o correggere una proposta: il suo `detection_id` più i campi che cambiano, almeno `applicability`;
   - falso positivo o doppione: non includerlo;
   - cartello mancante: un riquadro nuovo senza `detection_id`, con `bbox_xyxy` in pixel dell'immagine originale, `family` e `value_kph`;
   - numero non leggibile: `value_kph: null`; tipo non riconoscibile: `family: "unreadable"` con `notes`;
   - nessun cartello pertinente: `detections: []`.
5. **Salvare**, solo se autorizzati e dopo aver guardato l'immagine: la conferma è definitiva e diventa un dato di training al prossimo import, senza un'altra approvazione. Usare `save_frame_review` con `run_id` e `base_revision` esattamente come letti (`frame.save`) e `reviewer` con il nome dell'agente (per esempio `Claude Code`): il server lo salva come `ai/Claude Code`, tipo `ai`. Con la REST API indicare anche `reviewer_kind: "ai"`. Mai il nome di una persona.
6. Leggere l'esito:
   - riuscito: `review.revision`, `confirmed_count`, `next_unreviewed_frame_key`;
   - 409 `revision_conflict`: qualcuno ha salvato quel fotogramma nel frattempo. Rileggerlo con `get_frame`; non salvarci sopra senza autorizzazione;
   - 409 `stale_report`: il report è stato rigenerato. Fermarsi e avvisare Cristian;
   - 409 `human_review_exists`: il fotogramma è confermato da una persona. Lasciarlo com'è; `override_human_review: true` solo se Cristian lo chiede per quel fotogramma;
   - 400: valore non ammesso, il messaggio dice quale. Correggere e riprovare.
7. Annotarsi chiave, riquadri salvati e dubbi per il resoconto.

**Alla fine**
- `get_review_progress(route_id)` (REST: `GET /api/v1/routes/<route>`) dà i conteggi (anche `confirmed_by_ai` e `confirmed_by_human`), i sospetti con i motivi e i confermati con dubbi (`unknown_value`, `unreadable`, `unresolved_road`): servono per il resoconto della sezione 8, punto 5.
- `get_review_history(route_id, frame_key)` mostra tutte le revisioni di un fotogramma, con revisore, tipo, canale e ora.
- Le conferme AI sono definitive: non c'è una seconda approvazione di una persona. `nnslr import-annotations --from-db` le importa nel dataset come quelle umane, con la provenienza (`reviewer` `ai/<nome>`, `label_basis` `ai_review`, canale `rest` o `mcp`). Non importa i fotogrammi mai confermati, le conferme ritirate e le proposte del modello. Salvare non importa e non allena: import e training restano comandi separati che lancia Cristian, mai l'agente.
