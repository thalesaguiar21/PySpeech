import os


def _find_files(folder_path, extension):
    dt_files = []
    for dirpath, dirname, fnames in os.walk(folder_path):
        for fname in [f for f in fnames if f.lower().endswith('.' + extension)]:
            dt_files.append(os.path.join(dirpath, fname)
    print(f"Found {len(dt_files)} wav files!")
    return dt_files


def find_wav_files(folder_path):
    return _find_files(folder_path, "wav")


def find_txt_files(folder_path):
    return _find_files(folder_path, "txt")
