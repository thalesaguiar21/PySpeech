import os


def _find_files(folder_path, extension):
    dt_files = []
    for file_ in os.listdir(folder_path):
        if file_.lower().endswith("." + extension):
            dt_files.append(os.path.join(folder_path, file_))
    print(f"Found {len(dt_files)} wav files!")
    return dt_files


def find_wav_files(folder_path):
    return _find_files(folder_path, "wav")


def find_txt_files(folder_path):
    return _find_files(folder_path, "txt")
