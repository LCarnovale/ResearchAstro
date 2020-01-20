from astroquery.mast import Catalogs
import numpy as np
import matplotlib.pyplot as plt

candidates_path = 'candidates_out.txt'

# Get a target
from read_cdips import FitsTable
ft = FitsTable(6, 1, 1, 0)
ra = ft.RA
dec = ft.DEC

radius = 0.2 # degrees

# ra_range = 0.01
# dec_range = 0.01
# ra_min = ra - ra_range; ra_max = ra + ra_range
# dec_min = dec - dec_range; dec_max = dec + dec_range

# query = {
#     'ra': [ra_min, ra_max],
#     'dec': [dec_min, dec_max]
# }
# query mast
print("Querying...")
# t = Catalogs.query_object(f"{ra} {dec}", catalog="TIC", radius=radius)
t = Catalogs.query_criteria(1000, 0, catalog='TIC')
print("Found", len(t), "objects")
