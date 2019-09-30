import os


def _find_files(folder_path, extension):
    dt_files = []
    for dirpath, dirname, fnames in os.walk(folder_path):
        for fname in fnames:
            if fname.lower().endswith('.' + extension.lower()):
                dt_files.append(os.path.join(dirpath, fname))
    print(f"Found {len(dt_files)} {extension} files!")
    return dt_files


def find_wav_files(folder_path):
    return _find_files(folder_path, "wav")


def find_txt_files(folder_path):
    return _find_files(folder_path, "txt")
