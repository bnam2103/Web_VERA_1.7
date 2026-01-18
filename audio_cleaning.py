import shutil
from datetime import datetime, timedelta
from pathlib import Path

def cleanup_old_tts(days=3):
    cutoff = datetime.now() - timedelta(days=days)
    base = Path("tts_outputs")

    for folder in base.iterdir():
        if folder.is_dir():
            folder_date = datetime.strptime(folder.name, "%Y-%m-%d")
            if folder_date < cutoff:
                shutil.rmtree(folder)