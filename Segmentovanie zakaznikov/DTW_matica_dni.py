import pandas as pd
import numpy as np
from tslearn.metrics import dtw
import os #práca so súbormi (kontrola existencie)
import time #načasovanie a hodnotenie pokroku

#načítanie 15-min denných vektorov
vektor_den = pd.read_csv("daily_15min_vectors.csv")
vektor_den["day"] = pd.to_datetime(vektor_den["day"])
vektor_den.set_index(["dataid", "day"], inplace=True)
vektor_den = vektor_den.sort_index()

X = vektor_den.values
riadky_n = X.shape[0]

#ak už je čiastkova matica, tak začne z toho bodu, aby sa zabránilo opätovnému prepočítaniu všetkého
ciast_matica = "dtw_matrix_days_partial.npy"
start_index_file = "dtw_matrix_start_index.txt"
#kontrola, či už je uložená čiastočná matica
if os.path.exists(ciast_matica):
    dtw_matrix = np.load(ciast_matica)
    with open(start_index_file, "r") as f:
        start_i = int(f.read().strip())
    print(f"pokračovanie od riadku {start_i} / {riadky_n}")
else:
    dtw_matrix = np.zeros((riadky_n, riadky_n)) #prázdna symetrická matica vzdialeností
    start_i = 0
    print("výpočet od začiatku")

#nastavenie a zobrazenie času zostávajúceho do konca výpočtu matice (pre praktickosť)
start_time = time.time()
times = []

#výpočet DTW vzdialeností medzi každými dvomi dennými vektormi
for i in range(start_i, riadky_n):
    iter_start = time.time()

    for j in range(i + 1, riadky_n):
        vzdialenost = dtw(X[i], X[j]) #DTW medzi dňami i a j (96 prvkov)
        dtw_matrix[i, j] = vzdialenost
        dtw_matrix[j, i] = vzdialenost #symetrické

    iter_end = time.time()
    elapsed = iter_end - iter_start
    times.append(elapsed)

    #uloženie a print každých 10 iterácií
    if i % 10 == 0 or i == riadky_n - 1:
        np.save(ciast_matica, dtw_matrix)
        with open(start_index_file, "w") as f:
            f.write(str(i + 1))

        avg_time = np.mean(times[-10:])  #priemerný čas za posledných 10
        remaining = (riadky_n - i - 1) * avg_time
        mins = remaining / 60
        print(f"riadok {i+1}/{riadky_n} | zostáva cca: {mins:.1f} min")

#koneiec a uloženie
np.save("dtw_matrix_days.npy", dtw_matrix)
vektor_den.reset_index()[["dataid", "day"]].to_csv("dtw_index_map.csv", index=False)
