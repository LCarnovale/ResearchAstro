import numpy as np

files = [
    r"cdips-s0006_curl.sh",
    r"cdips-s0007_curl.sh",
    r"cdips-s0008_curl.sh",
    r"cdips-s0009_curl.sh",
    r"cdips-s0010_curl.sh",
    r"cdips-s0011_curl.sh",
]

out_path = 'g_ids.csv'
s = "source_id\n"

print("source_id")
for fname in files:
    with open(fname, 'r') as f:
        for line in f:
            if 'curl' not in line: continue
            l = line.split('--output')[1]
            l = l.split()[0][:-5] # Trim the .fits extension
            gaiaid = l.split('hlsp_cdips_tess_ffi_gaiatwo')[1]
            gaiaid = gaiaid.split('-')[0]
            s += gaiaid + '\n'
