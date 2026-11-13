# Consegna a Paperclip: unità internazionali per Speed Limit Vision

Data: 2 novembre 2026. Documento di proposta e consegna; implementazione non eseguita.

## Obiettivo

La rete deve leggere il numero scritto sul cartello. Il runtime Sunny deve interpretarne l'unità usando il paese ricavato dal GPS e pubblicare il limite fisico in m/s.

Il contratto resta:

```python
sm['speedVisionSP'].speedLimit  # m/s; 0 quando il limite non è disponibile
```

Il resolver riceve un valore già normalizzato, come accade per `carStateSP.speedLimit` e `liveMapDataSP.speedLimit`. Il display continua a seguire `is_metric`.

## Repository e riferimenti verificati

| Componente | Repository e riferimento esaminato |
|---|---|
| Dataset, training e portale | `cristianku/sunnypilot.speedlimits`, `main`, `936491a340ff97fda947d4ffcc1a68ef86510cef` |
| Runtime aggiunto | `cristianku/sunnypilot`, `speed-limits`, `8d37cf946ac8088ee503a60ba755bcb5a2445f77` |
| Sunny originale | `sunnypilot/sunnypilot`, `master`, `a86979d2b2b176413a7f3e5cccf8c807596a7def` |
| Parser della mappa | `pfeiferj/mapd`, versione `v1.12.0`, `46cd71ade6f630f1564c83bf1763f9d949a9ff30` |

Verificare nuovamente i riferimenti prima dell'implementazione. I branch sopra identificano il codice esaminato: non autorizzano cambi di branch, reset, pull o sovrascritture. Cristian richiede di preservare il lavoro esistente e lavorare su `main` salvo sua richiesta esplicita diversa.

## Fatti accertati

1. `speedvisiond` e la sorgente `speedVisionSP` sono aggiunte del nostro branch `speed-limits`. Il master Sunny esaminato usa le sorgenti auto/CAN e mappa.
2. NNLC significa Neural Network Lateral Control e riguarda il controllo dello sterzo. Non è una rete di lettura dei limiti. Non basare il lavoro sull'ipotesi di una sorgente stock `modelV2.speedLimit`.
3. La pipeline di training costruisce oggi classi come `maximum_speed:30` dal campo `value_kph`. Il contratto esportato conserva queste classi senza una distinzione dell'unità.
4. `vision.class_speed()` estrae il numero e lo interpreta come km/h. Anche `SpeedLimitMemory` e le sue tabelle di limiti impliciti lavorano attualmente in km/h.
5. `speedvisiond.py`, riga 130 del riferimento esaminato, pubblica sempre:

   ```python
   msg.speedVisionSP.speedLimit = float(speed_limit * CV.KPH_TO_MS)
   ```

6. `_get_from_vision()` nel resolver usa direttamente questo campo: non corregge l'unità originale.
7. Il paese della memoria viene inizializzato da `OsmLocationName`, con fallback `CH`. Non segue il GPS; anche paesi non presenti in `IMPLICIT_LIMITS`, fra cui `US` e `GB`, ricadono su `CH`.

### Riproduzione effettuata

Con l'uscita del classificatore controllata sulla classe esistente `maximum_speed:30`, usando le funzioni reali di decodifica, memoria, pubblicazione, selezione del resolver e aggiornamento del valore UI:

| Passaggio | Risultato attuale |
|---|---|
| Numero confermato su due frame | `30` |
| Limite pubblicato | `8.3333333333 m/s` |
| Limite ricevuto dal resolver | `8.3333333333 m/s` |
| HUD impostato in mph | `19 mph`, arrotondato da circa `18.64` |
| Valore corretto per un cartello di 30 mph | `13.4112 m/s` |

La prova usava output delle reti e messaggistica simulati: non dimostra il riconoscimento su immagini americane né il comportamento di una build installata sul comma. Il modello agganciato al branch non contiene la classe `25`; l'esempio di 25 mph è quindi un caso di conversione da testare, non un riconoscimento già dimostrato.

## Come funziona la sorgente mappa

Nel parser Go `ParseMaxSpeed()` di `mapd`:

- un valore senza suffisso, oppure con `kph`, `km/h`, `kmh`, viene interpretato come km/h;
- il suffisso `mph` applica il fattore `0.44704`;
- il risultato è in m/s.

La conversione avviene preparando il database offline, in `generate_offline.go`. Durante la guida `mapd` sceglie il tratto stradale e scrive il valore normalizzato in `MapSpeedLimit`. Sunny lo legge in `OsmMapData.get_current_speed_limit()` e lo assegna a `liveMapDataSP.speedLimit` senza altre conversioni.

Il modello da imitare è la normalizzazione prima del messaggio. La sorgente mappa conosce l'unità dal tag OSM; la vision deve ottenerla dal proprio contesto del paese.

## Soluzione proposta

### 1. Paese dal GPS e dai confini amministrativi offline

Usare il GPS del dispositivo (`gpsLocationExternal` oppure `gpsLocation`, selezionato dal helper Sunny esistente) per ricavare un contesto paese/unità.

Il lookup usa coordinate e confini amministrativi offline, senza richiedere la mappa delle strade o una connessione Internet. Per i paesi supportati mantenere una tabella esplicita delle unità: per esempio `GB` e `US` usano mph, `CH` e `FR` km/h. Paese non riconosciuto significa unità sconosciuta, non Svizzera o km/h come fallback automatico.

Precisazione del vincolo discusso: questo disegno ammette il contesto geografico GPS/paese per interpretare l'unità. `speedvisiond` e `SpeedLimitMemory` continuano a non leggere limiti, nomi delle strade, incroci o altri campi stradali di `mapd`/`liveMapDataSP`. La fusione dei limiti mappa e vision resta nel resolver.

Esiste già una logica offline nel progetto, `src/nnslr_tools/gps.py` (`country_of`, `world_place`) e `scripts/make_country_borders.py`. È un riferimento da valutare, non un modulo da importare interamente nel runtime: i dati usati dal portale hanno copertura e precisione diverse e documentano errori vicino ai confini. Scegliere dati adatti al runtime e verificarne copertura e licenza prima di distribuirli.

### 2. Numero letto e normalizzazione separati

Conservare famiglia, numero scritto e confidenza prodotti dal classificatore. La lettura affidabile della scritta `MPH` non deve essere un requisito per funzionare: anche nel Regno Unito il paese permette di interpretare i cartelli numerici in mph.

Non è richiesta una nuova testa della rete per l'unità nella prima implementazione. Una lettura futura dell'unità esplicita può essere valutata come controllo aggiuntivo, con una politica definita per eventuali conflitti col paese.

Conversioni esatte:

```text
km/h → m/s: numero / 3.6
mph  → m/s: numero * 0.44704
```

Normalizzare ogni nuova lettura prima che entri nella memoria operativa. Memoria, confronti e limite pubblicato devono usare m/s. La conversione riguarda il numero del cartello accettato, non un vecchio numero ricordato da reinterpretare quando cambia paese.

### 3. File runtime interessati

Percorsi relativi al repository `cristianku/sunnypilot`:

| File | Intervento proposto |
|---|---|
| `openpilot/sunnypilot/speed_limits/speedvisiond.py` | Acquisire il contesto paese GPS, passarne l'unità alla decodifica e pubblicare il limite già in m/s. Sostituire l'assunzione fissa della riga 130. |
| `openpilot/sunnypilot/speed_limits/vision.py` | Separare numero scritto e limite fisico in `class_speed`/`read_signs`; normalizzare prima della memoria; adeguare `SpeedLimitMemory`, confronti e costanti. |
| Helper e risorsa dei confini nel modulo `speed_limits` | Implementare il lookup paese/unità con dipendenze minime. Nome e suddivisione dei file sono da scegliere dopo aver esaminato le convenzioni del repository. |
| `openpilot/sunnypilot/speed_limits/tests/` | Test di conversione, memoria, contesto GPS e pubblicazione. |

Il publisher dovrà ricevere un risultato già normalizzato, per esempio:

```python
msg.speedVisionSP.speedLimit = float(speed_limit_mps)
```

Il resolver mantiene il contratto in m/s. Non aggiungere una conversione paese/unità dentro `_get_from_vision()`.

### 4. Adeguare anche limiti impliciti e log

Cambiare soltanto il moltiplicatore finale non basta: `IMPLICIT_LIMITS`, parcheggi, garage, zone e impostazioni dei rilasci hanno oggi una semantica in km/h. Convertire le costanti secondo la loro unità documentata, senza trattarle automaticamente come mph quando il paese è `US` o `GB`.

L'unità del paese e la legge sui limiti impliciti sono due problemi distinti. Sapere che siamo negli USA determina l'unità dei numeri, ma non basta per dedurre il limite legale di ogni strada o Stato. Dove mancano regole verificate per la giurisdizione, evitare il fallback ai limiti svizzeri e rappresentare il limite implicito come sconosciuto.

Preservare il significato dei campi cereal esistenti. `speedLimit` resta in m/s. `rawSignSpeed` è documentato in km/h: non cambiarlo silenziosamente in m/s o nel numero grezzo. Conservare tale semantica oppure proporre separatamente un'estensione additiva dei log per numero, unità e paese, con compatibilità dei vecchi messaggi.

## GPS incerto e attraversamenti di confine

Definire e testare questi comportamenti prima di considerare completa l'implementazione:

- una posizione rumorosa vicino al confine non deve alternare la conversione km/h/mph tra frame;
- valutare validità, età e accuratezza del fix e l'incertezza dei confini;
- con unità sconosciuta non accettare una nuova lettura numerica come se fosse km/h;
- un limite fisico già memorizzato non cambia scala al cambio paese;
- al cambio di giurisdizione rivalutare i limiti impliciti e definire esplicitamente la validità del limite precedentemente ricordato;
- un segnale di fine limite deve conservare il proprio significato anche quando non è disponibile una nuova regola implicita.

Soglie di stabilizzazione, conservazione del paese durante brevi perdite GPS e invalidazione della memoria restano scelte da motivare con replay e casi di confine: non sono state concordate soglie numeriche in questa conversazione.

Attenzione al comportamento già esistente del resolver/SLA: pubblicare `0` non dimostra da solo che siano stati dimenticati `speedLimitLast` e l'ultimo target. Verificare la catena completa prima di dichiarare risolto il caso paese sconosciuto/cambio confine. Se occorre un intervento sul consumo dello stato, documentarlo esplicitamente; l'unità sul messaggio rimane comunque m/s.

## Dataset e modelli

Le annotazioni devono distinguere il numero scritto dall'unità. Un cartello americano «25» va rappresentato come numero `25`, unità `mph`; `11.176 m/s` è il valore fisico derivato.

- `value_kph` è un nome legacy ambiguo: prevedere una migrazione esplicita e compatibile di annotazioni, API, import e contratti del modello.
- Conservare le classi del numero scritto con l'unità separata nei dati; non rinominare automaticamente un cartello da «25» a «40» per correggere l'unità.
- Il default del paese/unità può essere proposto dal GPS di ogni frame, con correzione nel review. Una route può attraversare un confine: una sola unità globale della route non è sufficiente.
- Dichiarare e versionare la semantica delle classi nei contratti del modello, evitando di confondere un modello legacy con uno addestrato sul numero scritto.
- L'unità corretta non garantisce il riconoscimento dei cartelli americani: servono classi numeriche, immagini e validazione adeguate.
- Il filtro paese con conteggi nelle schede training e nella pipeline, proposto nel contesto iniziale, è un'attività offline collegata. Il filtro da solo non garantisce il bilanciamento: verificare campionamento e risultati separati per paese, mantenendo separati train/validation/test per route o area.

Distinguere inoltre famiglia e applicabilità del cartello. La targhetta gialla «25 MPH» sotto un'uscita è una velocità consigliata per l'uscita: non va promossa automaticamente a limite massimo della carreggiata principale solo perché numero e unità sono leggibili.

## Criteri di accettazione

| Caso | Risultato richiesto |
|---|---|
| Numero 30, paese FR | `speedVisionSP.speedLimit ≈ 8.3333333333 m/s` |
| Stesso numero, paese GB o US | `≈ 13.4112 m/s` |
| Numero 25, unità mph, test del convertitore | `≈ 11.176 m/s` |
| Numero 50, unità mph | `≈ 22.352 m/s` |
| Cambio di `is_metric` | Valore pubblicato e selezionato dal resolver invariato; cambia la presentazione |
| GPS assente, scaduto o paese non coperto | Nessuna nuova interpretazione automatica in km/h; gestione dello stato esplicita e testata |
| Oscillazione GPS vicino al confine | Nessuna alternanza incontrollata dell'unità |
| Cambio paese confermato | Nessuna reinterpretazione numerica del limite vecchio; regole implicite aggiornate |
| US/GB senza regola implicita verificata | Nessun fallback silenzioso a `CH` |
| Limiti impliciti europei e parcheggi | Valori fisici invariati dopo la migrazione a m/s |
| Log vecchi e nuovi | Nessun cambio silenzioso di semantica dei campi cereal esistenti |
| Policy Map Data + Vision / Vision Only | Test dell'intero percorso, inclusi valori precedenti e SLA; nessuna seconda conversione nel resolver |

Per i test sulle unità usare la stessa lettura controllata sotto paesi diversi; poi verificare separatamente l'inferenza su immagini reali europee, britanniche e americane. Riportare distintamente test di codice, replay, inferenza e prove sul dispositivo.

## Consegna richiesta a Paperclip

1. Verificare i sorgenti e produrre un piano minimo per runtime e compatibilità dei dati.
2. Rendere esplicite le decisioni ancora aperte: dati dei confini, validità GPS, transizioni e storia del resolver/SLA.
3. Preparare modifiche e test nel perimetro autorizzato, con elenco dei file, risultati e limiti della verifica.

La preparazione di questo documento non ha modificato il codice applicativo. Non avviare import, training, pubblicazioni di modelli, servizi o deployment sulla base di questa sola consegna. Installazione, aggiornamento e riavvio del comma restano operazioni eseguite da Cristian, salvo sua richiesta esplicita per la specifica operazione. Non accendere Serverone-ai e non contattare worker dichiarati indisponibili.

## Riferimenti primari

- [Runtime: decodifica e memoria](https://github.com/cristianku/sunnypilot/blob/8d37cf946ac8088ee503a60ba755bcb5a2445f77/openpilot/sunnypilot/speed_limits/vision.py#L545).
- [Runtime: publisher e conversione attuale](https://github.com/cristianku/sunnypilot/blob/8d37cf946ac8088ee503a60ba755bcb5a2445f77/openpilot/sunnypilot/speed_limits/speedvisiond.py#L95).
- [Resolver vision](https://github.com/cristianku/sunnypilot/blob/8d37cf946ac8088ee503a60ba755bcb5a2445f77/openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_resolver.py#L120).
- [Schema SpeedVisionSP e unità dei log](https://github.com/cristianku/sunnypilot/blob/8d37cf946ac8088ee503a60ba755bcb5a2445f77/openpilot/cereal/custom.capnp#L488).
- [Training: costruzione della classe](https://github.com/cristianku/sunnypilot.smartspeed/blob/936491a340ff97fda947d4ffcc1a68ef86510cef/src/nnslr_tools/training.py#L100).
- [GPS e confini nel progetto offline](https://github.com/cristianku/sunnypilot.smartspeed/blob/936491a340ff97fda947d4ffcc1a68ef86510cef/src/nnslr_tools/gps.py#L192).
- [Mapd: conversione del tag](https://github.com/pfeiferj/mapd/blob/46cd71ade6f630f1564c83bf1763f9d949a9ff30/speed_limit.go#L8).
- [Mapd: conversione nella generazione offline](https://github.com/pfeiferj/mapd/blob/46cd71ade6f630f1564c83bf1763f9d949a9ff30/generate_offline.go#L152).
- [Sunny: lettura MapSpeedLimit](https://github.com/sunnypilot/sunnypilot/blob/a86979d2b2b176413a7f3e5cccf8c807596a7def/openpilot/sunnypilot/mapd/live_map_data/osm_map_data.py#L46).
- [Sunny: pubblicazione liveMapDataSP](https://github.com/sunnypilot/sunnypilot/blob/a86979d2b2b176413a7f3e5cccf8c807596a7def/openpilot/sunnypilot/mapd/live_map_data/base_map_data.py#L45).
- [Regno Unito: segnaletica e unità mph](https://www.gov.uk/government/publications/know-your-traffic-signs/speed-limit-signs).
- [Velocità consigliate sulle uscite: manuale NJTA, figura 6-1](https://www.njta.gov/wp-content/uploads/legacy/1422/dm_section-6-2015_signing_striping-new.pdf).
