from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
WAV_DIR = DATA_DIR / "wav"
FLAC_DIR = DATA_DIR / "flac"

if __name__ == '__main__':
    print(PROJECT_DIR)
    print(RAW_DIR)
