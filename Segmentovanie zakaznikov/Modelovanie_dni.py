import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

#načitanie dát
matica_DTW = np.load("dtw_matrix_days.npy")
index_map = pd.read_csv("dtw_index_map.csv")
vektory = pd.read_csv("daily_15min_vectors.csv")

#úprava a synchronizacia indexov
index_map["day"] = pd.to_datetime(index_map["day"]).dt.date
vektory["day"] = pd.to_datetime(vektory["day"]).dt.date
vektory.set_index(["dataid", "day"], inplace=True)
vektory.columns = vektory.columns.astype(int)

pairs = list(zip(index_map["dataid"], index_map["day"])) #spojenie dataid a dňa do zoznamu párov
pairs = [p for p in pairs if p in vektory.index]  #len tie páry, ktoré existujú vo vektoroch
chose = index_map.apply(lambda r: (r["dataid"], r["day"]) in pairs, axis=1)
common_indx = index_map[chose].reset_index(drop=True)
_matica_f = matica_DTW[chose.values][:, chose.values] #podmatica DTW len pre tieto spoločné dni
X15 = vektory.loc[pairs].values #hodnoty pre výbrane dni

#rozsahu počtu zhlukov
zluk_range = range(2, 18)
metrics = {"Silhouette": [], "Nenormalizovaná WCBCR": [], "WCBCR": []}
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']

for k in zluk_range:
    print(f"zhlukovanie na {k} zhlukov")
    model = KMedoids(n_clusters=k, metric="precomputed", init="k-medoids++", random_state=858)
    labels = model.fit_predict(_matica_f)

    #vypočet metrik
    sil_score = silhouette_score(_matica_f, labels, metric="precomputed")
    metrics["Silhouette"].append(sil_score)
    #sučet DTW vzdialenosti kazdeho bodu k jeho medoidu
    within = sum(_matica_f[np.where(labels == i)[0], model.medoid_indices_[i]].sum()
        for i in range(k))
    #sučet DTW vzdialenosti medzi medoidmi
    between = sum( _matica_f[i, j] for i in model.medoid_indices_ for j in model.medoid_indices_ if i < j)
    wcbcr = within / between if between else np.inf #ak between=0, vráti nekonečno namiesto deleniu nulou
    metrics["Nenormalizovaná WCBCR"].append(wcbcr)
    #normalizována WCBCR, použivame priemerne hodnoty
    avg_within = within / len(X15)
    avg_between = between / (k*(k-1)/2) if k > 1 else 1
    norm_wcbcr = avg_within / avg_between
    metrics["WCBCR"].append(norm_wcbcr)
    print(f"Silhouette: {sil_score:.3f}, WCBCR: {wcbcr:.2f}, norm-WCBCR: {norm_wcbcr:.3f}")

    #triedenie zhlukov podla priemernej spotreby
    zhluk_spotreba = {i: X15[labels==i].mean() for i in range(k)}
    sorted_ids = sorted(zhluk_spotreba, key=zhluk_spotreba.get)
    mapping = {old:new for new, old in enumerate(sorted_ids)}
    sortovane = np.array([mapping[l] for l in labels])

    #rozdelenie dní podľa zhlukov
    print("Rozdelenie dni:")
    for zhlk, poc_dni in pd.Series(sortovane).value_counts().sort_index().items():
        print(f"  Zhluk {zhlk}: {poc_dni} dni")

    #uloženie iba pre 6 zhlukov (lebo to bola optimálna hodnota)
    if k == 6:
        final_boss = common_indx.copy()
        final_boss["cluster"] = sortovane
        final_boss.to_csv("daily_clusters_6.csv", index=False)

    #vykreslenie priemernych profilov (agregovane do hodin)
    X_hodina = X15.reshape(-1, 24, 4).sum(axis=2)
    avg_profiles = [X_hodina[sortovane==zhlk].mean(axis=0) for zhlk in range(k)]
    plt.figure(figsize=(12, 6))
    for zhlk, prof in enumerate(avg_profiles):
        plt.plot(prof, label=f"Zhluk {zhlk}", color=colors[zhlk % len(colors)])
    plt.xticks(
        ticks=np.arange(0, 24, 3),
        labels=[f"{h:02d}:00" for h in range(0, 24, 3)]
    )
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.title(f"Priemerne profily - k={k}")
    plt.xlabel("čas dna")
    plt.ylabel("Spotreba (kWh)")
    plt.grid(True, which='both', axis='y', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


# vykreslenie metrik
for name, vals in metrics.items():
    plt.figure(figsize=(8, 4))
    plt.plot(list(zluk_range), vals, marker='o', color='green')
    plt.title(name)
    plt.xlabel("počet Zhlukov")
    plt.ylabel("hodnota")
    plt.grid(True)
    plt.tight_layout()
    plt.show()