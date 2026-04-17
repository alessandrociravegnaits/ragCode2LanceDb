"""
RAG Memory Manager per Copilot - powered by LanceDB
====================================================
Installazione:
    pip install lancedb sentence-transformers pyarrow

Uso:
    python memory_managerLanceDb.py   <- dalla cartella progetto
    oppure alias: rag

Struttura attesa nel progetto:
    mio_progetto/
        .copilot-memory/
            auth.md
            database.md
            api-orders.md
"""

import glob
import hashlib
import os
import sys
from datetime import datetime

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

# Configurazione
MEMORY_FOLDER = ".copilot-memory"
DB_PATH = os.path.join(os.path.expanduser("~"), ".rag_copilot_lancedb")
MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_TOP_K = 4
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_CONTEXT_CHARS = 1800

# Inizializzazione
print("Caricamento modello embedding...", end=" ", flush=True)
embedder = SentenceTransformer(MODEL_NAME)
EMBED_DIM = embedder.get_sentence_embedding_dimension()
TABLE_NAME = f"sessioni_v3_{MODEL_NAME.split('/')[-1].replace('-', '_').replace('.', '_')}"
print("ok")

db = lancedb.connect(DB_PATH)

schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("source_id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
    pa.field("progetto", pa.string()),
    pa.field("titolo", pa.string()),
    pa.field("data", pa.string()),
    pa.field("hash", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("chunk_total", pa.int32()),
])

table = db.create_table(TABLE_NAME, schema=schema, exist_ok=True)


# Helpers
def _hash_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return hashlib.md5(f.read().encode("utf-8")).hexdigest()


def _source_id(path: str) -> str:
    normalized = os.path.abspath(path).lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _where_eq(field: str, value: str) -> str:
    return f"{field} = {_sql_quote(value)}"


def _embed_many(texts: list[str]) -> list[list[float]]:
    vectors = embedder.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Chunking semplice a caratteri con overlap per recupero piu stabile."""
    cleaned = text.strip()
    if not cleaned:
        return []

    if chunk_size <= 0:
        return [cleaned]

    step = max(chunk_size - overlap, 1)
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start += step
    return chunks


def _get_rows_by_source(nome_progetto: str, source_id: str) -> list[dict]:
    where = f"{_where_eq('progetto', nome_progetto)} AND {_where_eq('source_id', source_id)}"
    try:
        return table.search(None).where(where).to_list()
    except Exception as exc:
        print(f"   Errore query sorgente ({source_id[:8]}): {exc}")
        return []


def _delete_rows_by_source(nome_progetto: str, source_id: str):
    where = f"{_where_eq('progetto', nome_progetto)} AND {_where_eq('source_id', source_id)}"
    try:
        table.delete(where)
    except Exception as exc:
        print(f"   Errore delete sorgente ({source_id[:8]}): {exc}")


def _truncate(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[troncato]"


def _render_progress_bar(current: int, total: int, width: int = 24) -> str:
    total = max(total, 1)
    current = max(min(current, total), 0)
    filled = int((current / total) * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _print_progress(prefix: str, current: int, total: int, detail: str = "") -> None:
    bar = _render_progress_bar(current, total)
    percent = int((max(min(current, total), 0) / max(total, 1)) * 100)
    suffix = f" {detail}" if detail else ""
    sys.stdout.write(f"\r{prefix} {bar} {percent:3d}% ({current}/{max(total, 1)}){suffix}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


# Core functions
def rileva_progetto() -> tuple[str, str]:
    """Rileva cartella e nome progetto dalla directory corrente."""
    cwd = os.getcwd()
    nome = os.path.basename(cwd)
    return cwd, nome


def indicizza_progetto(cartella: str, nome_progetto: str) -> int:
    """
    Scansiona .copilot-memory/ e indicizza i file MD nuovi o modificati.
    I documenti vengono salvati a chunk per migliorare la retrieval.
    """
    memory_dir = os.path.join(cartella, MEMORY_FOLDER)

    if not os.path.exists(memory_dir):
        os.makedirs(memory_dir)
        print(f"\nCreata cartella: {memory_dir}")
        print("Salva qui i riassunti in markdown per alimentarli nel RAG.")
        return 0

    md_files = sorted(set(
        glob.glob(os.path.join(memory_dir, "*.md"))
        + glob.glob(os.path.join(memory_dir, "**", "*.md"), recursive=True)
    ))
    # Escludi il file di avvio automatico se presente nella cartella memory
    md_files = [p for p in md_files if os.path.basename(p) != "99-inizioSessioneVettoriale.md"]

    if not md_files:
        print("Nessun file MD trovato in .copilot-memory/")
        return 0

    total_files = len(md_files)
    nuovi = 0
    aggiornati = 0
    saltati = 0
    chunks_totali = 0

    for processed_files, filepath in enumerate(md_files, start=1):
        titolo = os.path.basename(filepath)
        source_id = _source_id(filepath)
        file_hash = _hash_file(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            contenuto = f.read()

        if not contenuto.strip():
            print(f"   File vuoto ignorato: {titolo}")
            _print_progress("Indicizzazione", processed_files, total_files, f"{titolo} (vuoto)")
            continue

        existing_rows = _get_rows_by_source(nome_progetto, source_id)
        if existing_rows:
            old_hash = existing_rows[0].get("hash", "")
            if old_hash == file_hash:
                saltati += 1
                _print_progress("Indicizzazione", processed_files, total_files, f"{titolo} (invariato)")
                continue
            _delete_rows_by_source(nome_progetto, source_id)
            aggiornati += 1
        else:
            nuovi += 1

        chunks = _chunk_text(contenuto)
        if not chunks:
            print(f"   File senza contenuto utile: {titolo}")
            continue

        vectors = _embed_many(chunks)
        now_date = datetime.now().isoformat()[:10]
        total = len(chunks)

        rows = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
            row_id = f"{source_id}:{idx}"
            rows.append({
                "id": row_id,
                "source_id": source_id,
                "text": chunk,
                "vector": vector,
                "progetto": nome_progetto,
                "titolo": titolo,
                "data": now_date,
                "hash": file_hash,
                "chunk_index": idx,
                "chunk_total": total,
            })

        table.add(rows)
        chunks_totali += total
        _print_progress("Indicizzazione", processed_files, total_files, f"{titolo} ({total} chunk)")

    processati = nuovi + aggiornati
    print(
        f"Indicizzazione completata: {nuovi} nuovi, {aggiornati} aggiornati, "
        f"{saltati} invariati, {chunks_totali} chunk salvati (file trovati: {len(md_files)})."
    )
    return processati


def recupera_contesto(domanda: str, nome_progetto: str, n: int = DEFAULT_TOP_K) -> str:
    """Recupera i chunk semanticamente piu rilevanti per il progetto corrente."""
    totale = table.count_rows()
    if totale == 0:
        return "Nessuna sessione indicizzata ancora."

    query_vector = _embed_many([domanda])[0]
    where = _where_eq("progetto", nome_progetto)

    # Recupera piu candidati del necessario per gestire eventuali duplicati.
    search_limit = max(n * 4, n)

    try:
        risultati = table.search(query_vector).where(where).limit(search_limit).to_list()
    except Exception as exc:
        return f"Errore durante la ricerca vettoriale: {exc}"

    if not risultati:
        return (
            f"Nessun risultato per il progetto '{nome_progetto}'.\n"
            "Indicizza prima i file con l'opzione 2."
        )

    # Costruisci output con intestazione esplicita per Copilot e metadati di provenienza
    header = []
    header.append(f"# Contesto recuperato per progetto: {nome_progetto}")
    header.append("")
    header.append("USARE QUESTO BLOCCO COME FONTE PRIMARIA: se ci sono conflitti, privilegiare le informazioni qui riportate.")
    header.append("Quando citi, indica la sorgente come: [Titolo] chunk N/M.")
    header.append("")

    header.append("Chunk selezionati automaticamente:")
    for idx, row in enumerate(risultati[:n], start=1):
        titolo = row.get("titolo", "senza-titolo")
        chunk_index = row.get("chunk_index", "?")
        chunk_total = row.get("chunk_total", "?")
        distanza = row.get("_distance")
        score_text = f"score={distanza:.4f}" if isinstance(distanza, (float, int)) else "score=n/a"
        header.append(f"- [{idx}] {titolo} chunk {chunk_index}/{chunk_total} | {score_text}")
    header.append("")

    body = []
    for idx, row in enumerate(risultati[:n], start=1):
        titolo = row.get("titolo", "senza-titolo")
        data = row.get("data", "")
        chunk_index = row.get("chunk_index", "?")
        chunk_total = row.get("chunk_total", "?")
        distanza = row.get("_distance")
        source_id = row.get("source_id", "")
        score_text = f"score={distanza:.4f}" if isinstance(distanza, (float, int)) else ""

        body.append(f"--- [{idx}] {titolo} ({data}) | chunk {chunk_index}/{chunk_total} | {score_text}")
        body.append(f"SOURCE_ID: {source_id}")
        body.append("")
        body.append(_truncate(row.get("text", "")))
        body.append("")

    return "\n".join(header + ["---"] + body).strip()


def _auto_select_chunk_count(question: str, nome_progetto: str, total_chunks: int) -> int:
    """Sceglie automaticamente un numero di chunk coerente con il salto degli score."""
    if total_chunks <= 0:
        return DEFAULT_TOP_K

    try:
        query_vector = _embed_many([question])[0]
        preview_limit = min(max(total_chunks, DEFAULT_TOP_K), 40)
        preview_results = table.search(query_vector).where(_where_eq("progetto", nome_progetto)).limit(preview_limit).to_list()
        distances = [r.get("_distance", 0.0) for r in preview_results if isinstance(r.get("_distance", 0.0), (float, int))]
        if len(distances) >= 2:
            diffs = [distances[i + 1] - distances[i] for i in range(len(distances) - 1)]
            max_gap = max(diffs)
            max_gap_idx = diffs.index(max_gap)
            recommended_n = max(1, max_gap_idx + 1)
            recommended_n = min(recommended_n, min(12, total_chunks))
            if recommended_n < 2 and total_chunks >= 3:
                recommended_n = 3
            return recommended_n
    except Exception:
        pass

    return min(DEFAULT_TOP_K, total_chunks)


def lista_sessioni(nome_progetto: str):
    """Mostra i file indicizzati per il progetto corrente (aggregati per sorgente)."""
    try:
        rows = table.search(None).where(_where_eq("progetto", nome_progetto)).to_list()
    except Exception as exc:
        print(f"\nErrore lettura indice: {exc}")
        return

    if not rows:
        print(f"\nNessun file indicizzato per '{nome_progetto}'.")
        return

    aggregati = {}
    for row in rows:
        source_id = row.get("source_id")
        if source_id not in aggregati:
            aggregati[source_id] = {
                "titolo": row.get("titolo", ""),
                "data": row.get("data", ""),
                "chunks": 0,
            }
        aggregati[source_id]["chunks"] += 1

    print(f"\nFile indicizzati per '{nome_progetto}' ({len(aggregati)}):\n")
    for _, value in sorted(aggregati.items(), key=lambda x: x[1]["titolo"].lower()):
        print(f"  - {value['titolo']} ({value['data']}) - {value['chunks']} chunk")


# Menu
def _stampa_menu(nome: str, cartella: str):
    print(f"""
+------------------------------------------------------+
|      RAG Memory Manager - LanceDB                    |
+------------------------------------------------------+
|  Progetto : {nome:<32}|
|  Cartella : ...{cartella[-29:]:<29}|
+------------------------------------------------------+
|  1. Inizio sessione (indicizza + recupera contesto)  |
|  2. Fine sessione   (indicizza nuovi MD)             |
|  3. Lista file indicizzati                           |
|  4. Esci                                              |
+------------------------------------------------------+
""")


def menu():
    cartella, nome = rileva_progetto()

    while True:
        _stampa_menu(nome, cartella)
        scelta = input("Scelta: ").strip()

        if scelta == "1":
            print("\nIndicizzazione MD in corso...")
            indicizza_progetto(cartella, nome)

            domanda = input("\nSu cosa lavori oggi? ").strip()
            if not domanda:
                print("Domanda vuota.")
            else:
                # Conta chunk disponibili per il progetto
                try:
                    rows_proj = table.search(None).where(_where_eq("progetto", nome)).to_list()
                    total_chunks = len(rows_proj)
                except Exception:
                    total_chunks = 0

                # Mostra solo l'informazione minima prima della selezione automatica
                print(f" - Chunk disponibili: {total_chunks}")

                # Selezione automatica dei chunk migliori con fallback sicuro
                recommended_n = _auto_select_chunk_count(domanda, nome, total_chunks)

                print(f"Raccomandazione automatica basata sugli score: {recommended_n} chunk")

                n = min(recommended_n, total_chunks) if total_chunks > 0 else recommended_n

                contesto = recupera_contesto(domanda, nome, n)

                # Salva automaticamente il contesto in root del progetto
                filename = "99-inizioSessioneVettoriale.md"
                out_path = os.path.join(cartella, filename)
                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(contesto)
                    print("\nContesto salvato in:", out_path)
                    print("Nota: questo file è nella root del progetto e non verrà indicizzato.")
                    print("Puoi ora allegare il file alla nuova chat Copilot.")
                except Exception as exc:
                    print(f"Errore salvataggio contesto su {out_path}: {exc}")

        elif scelta == "2":
            print("\nIndicizzazione nuovi MD...")
            n_file = indicizza_progetto(cartella, nome)
            if n_file == 0:
                print("Nessun file nuovo o aggiornato da indicizzare.")

        elif scelta == "3":
            lista_sessioni(nome)

        elif scelta == "4":
            sys.exit(0)
        else:
            print("Scelta non valida.")

        input("\nPremi INVIO per continuare...")


if __name__ == "__main__":
    menu()
