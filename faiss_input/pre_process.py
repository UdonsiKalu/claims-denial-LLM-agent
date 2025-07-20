import pandas as pd
import os

def convert_encoding(input_path, output_path, encodings=['utf-8', 'latin1', 'windows-1252']):
    for encoding in encodings:
        try:
            df = pd.read_csv(input_path, encoding=encoding)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Successfully converted {input_path} from {encoding} to UTF-8")
            return True
        except UnicodeDecodeError:
            continue
    return False

base_path = "/media/udonsi-kalu/New Volume/denials/denials/faiss_input/ncd_csv"
files = ['ncd_trkg.csv', 'ncd_trkg_bnft_xref.csv', 'ncd_bnft_ctgry_ref.csv', 'ncd_pblctn_ref.csv']

for file in files:
    input_file = os.path.join(base_path, file)
    output_file = os.path.join(base_path, f"utf8_{file}")
    if not convert_encoding(input_file, output_file):
        print(f"Failed to convert {file}")