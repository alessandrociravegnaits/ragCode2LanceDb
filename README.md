# memory_managerLanceDb

Repository standalone per il solo RAG memory manager basato su LanceDB.

## Scopo

Indicizza i file Markdown presenti in `.copilot-memory/` e recupera i chunk più rilevanti usando il modello di embedding configurato nello script.

## File principali

- `memory_managerLanceDb.py`: script principale
- `requirementsLance.txt`: dipendenze minime del progetto
- `.gitignore`: esclusioni per file locali, database e ambienti virtuali

## Uso rapido

```bash
pip install -r requirementsLance.txt
python memory_managerLanceDb.py
```

## Come usare i prompt

Il flusso è pensato per essere semplice:

1. Avvia lo script con `python memory_managerLanceDb.py`
2. Usa il menu `1` per aprire una sessione con contesto già pronto
3. Usa il menu `2` per chiudere la sessione e salvare i riassunti nuovi

### Prompt di apertura

Quando scegli il menu `1`, nel prompt scrivi una frase come questa:

```text
Usa il contesto 99-inizioSessioneVettoriale per riprendere il lavoro dal punto in cui eravamo:

Ora: descrivi cosa vuoi fare oggi
```

Questo serve a riprendere il lavoro dall’ultimo riepilogo salvato nella root del progetto.

### Prompt di chiusura

Quando scegli il menu `2`, scrivi nel prompt di chiusura qualcosa del tipo:

```text
Salva sempre i riassunti di sessione suddivisi per argomento in .copilot-memory/ nella root del progetto, in formato MD.
```

Questo flusso aggiorna i file Markdown già presenti in `.copilot-memory/` e salva il nuovo contesto in `99-inizioSessioneVettoriale.md`.

## Cosa fa il menu

- `1`: indicizza i file Markdown della cartella `.copilot-memory/` e recupera il contesto migliore per l’apertura sessione
- `2`: indicizza solo i nuovi o modificati Markdown della cartella `.copilot-memory/`
- `3`: mostra la lista dei file già indicizzati
- `4`: esce

## Struttura attesa

```text
progetto/
  .copilot-memory/
    note.md
    sessione.md
  memory_managerLanceDb.py
  requirementsLance.txt
  README_memory_managerLanceDb.md
```

## Comportamento

- indicizza tutti i file `.md` dentro `.copilot-memory/`
- riusa i file già indicizzati se non sono cambiati
- salva il contesto di apertura in `99-inizioSessioneVettoriale.md`
- usa LanceDB come archivio vettoriale locale
- usa il modello di embedding `BAAI/bge-large-en-v1.5`

## File da caricare nella repo

Se vuoi pubblicare solo questo progetto, i file essenziali sono:

- `memory_managerLanceDb.py`
- `requirementsLance.txt`
- `README_memory_managerLanceDb.md`
- `.gitignore`

Opzionali, se vuoi includere i prompt operativi:

- `promptApertura.txt`
- `prompChiusura.txt`

## Nota

Se cambi modello embedding o schema, conviene usare una tabella LanceDB separata per evitare di mescolare indici generati con configurazioni diverse.
