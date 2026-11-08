# Parere tecnico — perché il Brain non impara il prossimo limite

**Data:** 8 novembre 2026  
**Risposta a:** [2026-11-08-next-limit-advice-request.md](2026-11-08-next-limit-advice-request.md)  
**Ambito:** sola visione e segnali dell'automobile; OpenStreetMap può produrre etichette *offline* ma non entra mai nel Brain sul dispositivo.

## Diagnosi

**Il principale indiziato è l'ambiguità degli input, non la dimensione della rete.**

Sui frame di test in cui il prossimo numero è già confermato nell'input `signs`, il Brain bilanciato indovina **108/515 (circa 21%)** e risponde **«none» su 380/515**. Dunque il numero spesso c'è già, ma il modello non distingue bene **cartello ancora davanti** da **cartello appena superato o già in vigore**.

Il documento riporta inoltre **410/981** frame con un prossimo limite noto nel target ma cartello **non ancora letto** nell'input. In quei casi il modello non può ricavare con affidabilità le cifre da un'immagine a 256×144: chiedergli il valore esatto a priori rende il task parzialmente non osservabile.

I dati sono pochi **per evento**: **120 cambi di limite in train**, contro 2.840 frame positivi. Molti frame consecutivi non equivalgono a molti esempi indipendenti.

### Ipotesi, in ordine di evidenza

1. **#4 — Ambiguità tra input e target: forte evidenza diretta.** Un numero letto non porta con sé un'identità persistente del cartello e una fase affidabile di avvicinamento/superamento.
2. **#1 — Pochi dati: limite reale.** Solo 120 eventi train rendono difficile imparare transizioni rare e classi «none»/«ahead» ben calibrate.
3. **#2 — Bassa risoluzione: plausibile prima del rilevamento.** L'immagine 256×144 non risolve cifre lontane; però il fallimento sui 515 frame con numero già noto non si spiega solo con la risoluzione.
4. **#3 — Architettura insufficiente: non dimostrato.** Prima di cambiare MobileNetV3 o ingrandire la GRU, renderei osservabile il passaggio del cartello e correggerei il task.

## Le tre modifiche consigliate, in ordine

### 1. Fornire al Brain una storia temporale di ciascun cartello (priorità massima)

**Risolve soprattutto:** ipotesi #4.

Riutilizzare SignNet e il classificatore ad alta risoluzione per costruire un **track per cartello** attraverso i frame. Passare al Brain, per i candidati ancora attivi, informazioni compatte quali:

| Feature disponibile a bordo | Che cosa permette di imparare |
|---|---|
| Classe/valore e confidenza | Che cosa è stato letto |
| Distanza stimata e flag di validità | Se il cartello dovrebbe essere ancora davanti |
| Bounding box: posizione e dimensione | Geometria del cartello |
| Variazione dimensione nel tempo | Avvicinamento |
| Tempo dall'ultimo avvistamento, visibile/non visibile | Persistenza e scomparsa |
| Identità stabile del track (per associazione, non come numero semantico) | Distinguere due cartelli successivi con lo stesso valore |
| Odometria relativa al track | Evidenza del suo passaggio |

**Non** basta inviare solo un ID arbitrario: l'ID identifica il track, mentre le feature geometriche e temporali forniscono le informazioni utili alla rete.

Nel codice esiste già [`SignDistance`](https://github.com/cristianku/sunnypilot/blob/speed-limits/openpilot/sunnypilot/speed_limits/vision.py), che mantiene avvistamenti, dimensioni e odometria, ma non un vero tracking simultaneo di più cartelli. È il punto di partenza naturale.

La rete deve decidere da sola quando un segnale è prossimo, passato o irrilevante. **Nessuna regola hard-coded che scelga il limite al posto del modello.** Sono ammessi il tracking e il calcolo geometrico come preprocessing, purché la scelta finale rimanga neurale.

**Verifica:** isolare i 515 frame di test con numero corretto già confermato e controllare *accuracy sull'ahead*, falsi «none», falsi «ahead» e latenza del primo riconoscimento corretto. Un primo obiettivo sperimentale potrebbe essere >50% contro ~21%, **non una previsione garantita**; deve accompagnarsi a falsi positivi accettabili.

### 2. Allineare i target a ciò che il modello può sapere (seconda priorità)

**Risolve:** #4 e parte di #1.

L'attuale [`ahead_from_signs`](../src/nnslr_tools/brain_targets.py) insegna il numero nei 100 m prima del cartello, anche quando il lettore non l'ha ancora identificato. La classe «none» viene invece assegnata soltanto dove sono documentati 130 m senza segnali; la fascia intermedia resta fuori dalla loss.

Proverei una variante controllata del dataset:

- **Numero futuro:** supervisionarlo quando il corrispondente segnale è già leggibile/rilevato, mantenendo separata la valutazione dell'intero intervallo a 100 m.
- **Non osservato ≠ «none»:** quando la lettura è assente o la copertura è incerta, non insegnare automaticamente l'assenza di un limite.
- **Attraversamento:** concentrare esempi e loss sui secondi *prima, durante e dopo* il passaggio, non soltanto sui tanti frame stabili.
- **Bilanciamento per evento/track:** impedire che un unico cartello visibile a lungo domini il training.
- **Passaggio verificato:** controllare le etichette vicine allo zero. Senza marcatura manuale, il codice usa `PASSAGE_SECONDS = 0.5` dopo l'ultima box: è una stima, e può spostare proprio i target critici.

In valutazione bisogna mantenere **due metriche separate**: «limite corretto quando il segnale è già leggibile» e «limite anticipato correttamente entro la distanza necessaria per frenare». Non mascherare un peggioramento nella seconda metrica migliorando solo la prima.

**Verifica:** train/validation/test invariati, misura per **eventi indipendenti** oltre che per frame; quanti cambi corretti sono riconosciuti *prima* del passaggio, anticipo in metri, falsi «ahead», errore di distanza e prestazioni della speed head.

### 3. Soltanto se necessario: una piccola memoria per i cartelli (terza priorità)

**Risolve:** possibile #3, dopo aver reso osservabile il fenomeno.

Confrontare la GRU attuale con una **piccola GRU dedicata ai sign tracks** (per esempio hidden size 32), senza duplicare l'encoder visivo:

```text
 Camera → MobileNetV3 ───────────────┐
                                     │
 SignNet + Classifier → Sign tracks → Tiny GRU
                                     │
                      ┌──────────────┘
                      ▼
                 Brain GRU
                      │
             ┌────────┴─────────┐
             ▼                  ▼
       Speed to hold      Next limit + metres
```

La Tiny GRU costruisce una rappresentazione temporale di *cartello in avvicinamento / ancora visibile / probabilmente superato*; la GRU del Brain usa quella rappresentazione insieme al contesto e **continua a prendere tutte le decisioni**. Nessuna mappa e nessun motore di regole separato.

Con 120 eventi, più parametri potrebbero peggiorare la generalizzazione: questa è una **ablation da provare, non la prima implementazione**.

**Verifica:** stessi split per sito, stessi eventi, stesse feature e target; confrontare GRU unica vs due GRU per accuratezza evento, errori di distanza, falsi positivi, latenza e budget GPU a 4 Hz su comma 4.

## Un'ulteriore semplificazione da sperimentare

Nel [`brain_net.py`](../src/nnslr_tools/brain_net.py), `next_head` produce attualmente **201 classi**: 200 velocità e «none». Il termine `trust_next * onehot(number)` incoraggia il numero letto, ma **non impedisce** alla rete di selezionarne uno diverso.

Con pochi dati si può confrontare un output **seleziona un track già riconosciuto / nessun track** (con numero e distanza associati al track), anziché richiedere una classificazione autonoma su 201 valori. La selezione e la validità del track restano apprese; il valore numerico viene dal lettore già presente. Questo approccio non risolve da solo un cartello *non ancora letto* e richiede gestione dei candidati multipli.

## Piano minimo di esperimenti

1. **Baseline riproducibile:** conservare seed, checkpoint, split e misure attuali. Registrare per ogni evento: primo frame letto, primo frame predetto correttamente, punto di passaggio, falsi «ahead», errore in metri.
2. **Ablation A — input:** aggiungere track/visibilità/dimensione/distanza, senza cambiare la GRU, e misurare i 515 frame con numero già letto.
3. **Ablation B — target:** a parità di rete/input, correggere il mascheramento dei target prima della lettura e pesare gli eventi.
4. **Ablation C — architettura:** solo dopo A e B, confrontare la Tiny GRU o la selezione neurale dei track.
5. **Controllo reale:** replay su interi percorsi tenuti fuori dal training e poi prova in modalità osservazione, senza affidare a un modello non validato il comando automatico della velocità.

**Criterio di successo prioritario:** percentuale di cambi di limite riconosciuti correttamente **con sufficiente distanza di anticipo per la decelerazione**, non solo percentuale di frame corretti. Un cartello riconosciuto a 30–40 m può comunque arrivare troppo tardi per rallentare in sicurezza.

## Risposta alla domanda «Brain unico o rete dedicata?»

**Terrei il prossimo limite nel Brain come decisione finale.** Aggiungerei prima un front-end di *tracking* e feature temporali dei cartelli; se non basta, una piccola memoria specializzata che **alimenta** il Brain. Non farei un secondo decisore indipendente che applica limiti tramite regole.

**In sintesi:** prima rendere chiaro *quale cartello, dov'è e se è già passato*; poi migliorare le etichette; solo infine aumentare la capacità della rete.
