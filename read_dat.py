import pandas as pd
import numpy as np

source_file = r"Y:\Uni\ResearchStello\unzipped1\SG_US_step.dat"

output = r"Y:\Uni\ResearchStello\out.txt"

chunk_size = 256
df = pd.read_csv(source_file, header=0, delim_whitespace=True, chunksize=chunk_size)

with open(output, 'w') as f:
    f.write('id, mass, radius, Y_init, Feh_init, alpha, diff, overshoot, teff, lum\n')

# flag = True
n = 1
for chunk in df:
    duplicate_count = 0
    cols = np.array(chunk.columns.values.tolist())
    lines = []
    for i in range(len(chunk)):
        vals = chunk.iloc[i].values
        mask = (~np.isnan(vals))
        available_cols = cols[mask] # non-NaN column names
        available_row_data = vals[mask] # non-NaN row values

        # numax = available_row_data[available_cols == 'nu_max']
        # age = available_row_data[available_cols == 'age']
        mass = available_row_data[available_cols == 'M'][0]
        id = available_row_data[available_cols == 'id'][0]
        init_helium = available_row_data[available_cols == 'Y'][0]
        init_metallicity = available_row_data[available_cols == 'Z'][0]
        alpha = available_row_data[available_cols == 'alpha'][0]
        diffusion = available_row_data[available_cols == 'diffusion'][0]
        overshoot = available_row_data[available_cols == 'overshoot'][0]
        # undershoot = available_row_data[available_cols == 'undershoot'][0]
        # ev_state = available_row_data[available_cols == 'ev_stage'][0]
        radius = available_row_data[available_cols == 'radius'][0]
        teff = available_row_data[available_cols == 'Teff'][0]
        # log_g = available_row_data[available_cols == 'log_g']
        luminosity = available_row_data[available_cols == 'L'][0]
        # acoustic_cutoff = available_row_data[available_cols == 'acoustic_cutoff']
        # period_spacing = available_row_data[available_cols == 'delta_Pg_asym']
        # core_hydrogen_frac = available_row_data[available_cols == 'X_c']

        new_line = f"{id}, {mass}, {radius}, {init_helium}, {init_metallicity}, {alpha}, {diffusion}, {overshoot}, {teff}, {luminosity}"
        if not lines or lines[-1] != new_line:
            lines.append(new_line)
        else:
            duplicate_count += 1
    print("Writing chunk %d... (%d duplicates) "%(n, duplicate_count))
    n += 1
    with open(output, 'a') as f:
        s = '\n'.join(lines) + '\n'
        f.write(s)

print(f"Done {n} chunks of size {chunk_size}")
