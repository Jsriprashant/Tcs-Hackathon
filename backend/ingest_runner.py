#!/usr/bin/env python3
"""Ingestion runner that writes all output to a file."""
import sys
import os
from pathlib import Path
from datetime import datetime

# Setup paths
backend = Path(__file__).parent.absolute()
os.chdir(backend)
sys.path.insert(0, str(backend))

# Output file
log_file = backend / "ingestion_output.log"

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout
sys.stdout = Logger(log_file)
sys.stderr = sys.stdout

print(f"Ingestion started at: {datetime.now()}")
print(f"Log file: {log_file}")
print("=" * 60)
print()

try:
    # Import and run
    from data.document_loader import (
        load_all_documents, 
        RAW_DATA_PATH, 
        get_chroma_client,
        COLLECTIONS,
        get_vectorstore
    )
    
    print(f"Data source: {RAW_DATA_PATH}")
    print(f"ChromaDB: {get_chroma_client()}")
    print()
    
    # Count files
    csv_files = list(RAW_DATA_PATH.rglob("*.csv"))
    md_files = list(RAW_DATA_PATH.rglob("*.md"))
    txt_files = list(RAW_DATA_PATH.rglob("*.txt"))
    
    print(f"Files found:")
    print(f"  CSV: {len(csv_files)}")
    print(f"  MD:  {len(md_files)}")
    print(f"  TXT: {len(txt_files)}")
    print()
    
    print("Starting ingestion...")
    print("-" * 40)
    
    results = load_all_documents()
    
    print()
    print("=" * 60)
    print("INGESTION RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    # Verify
    print()
    print("Verifying collections...")
    for name, coll in COLLECTIONS.items():
        try:
            vs = get_vectorstore(coll)
            count = vs._collection.count()
            print(f"  {name}: {count} vectors")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Finished at: {datetime.now()}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    sys.stdout.log.close()
