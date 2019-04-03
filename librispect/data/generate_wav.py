"""
Extract the downloaded file and convert to .wav
"""

import os
import tarfile
import deepspectrograminversion as dsi
import glob
from sox import Transformer
import shutil

dsi.paths.WAV_DIR.mkdir(parents=True, exist_ok=True)
dsi.paths.FLAC_DIR.mkdir(parents=True, exist_ok=True)


def to_wav(extracted_dir, destination_dir):
    """this function convert the trans.txt files in extracted directories to .wav files 
        and assign them into a new destination as user intended
    """

    for path in glob.glob(extracted_dir + "/**/*.flac", recursive=True):
        print("generating file...")
        flac_file = path
        seq_id = os.path.split(path)[-1].split('.')[0]
        wav_file = destination_dir + "/" + (seq_id + ".wav")
        if not os.path.exists(wav_file):
            print("forming wav file. ID: " + seq_id)
            Transformer().build(flac_file, wav_file)

    print("TRANSCRIPTION DONE")


def delete_flacs():
    print("Start deleting flac files...")
    shutil.rmtree(dsi.paths.FLAC_DIR)
    print("DELETION COMPLETE")


def extract(destination, archive):
    """This function extract files in archive to a directory"""
    with tarfile.open(archive) as tar:
        tar.extractall(destination)


def conversion():
    """This function runs extract and convert"""
    for path in glob.glob(str(dsi.paths.RAW_DIR / "*.tar.gz")):
        extract(str(dsi.paths.FLAC_DIR), path)
    to_wav(str(dsi.paths.FLAC_DIR), str(dsi.paths.WAV_DIR))

if __name__ == "__main__":
    conversion()
    delete_flacs()
