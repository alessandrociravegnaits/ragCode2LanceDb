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

## Nota

Se cambi modello embedding o schema, conviene usare una tabella LanceDB separata per evitare di mescolare indici generati con configurazioni diverse.
