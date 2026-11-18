#1) matica vzdialenosti pre týždne
import pandas as pd
import numpy as np
from tslearn.metrics import dtw
import matplotlib.pyplot as plt
import seaborn as sns

#načitanie suboru vektorov a suboru zhlukov pre denné vektory
vektory = pd.read_csv("daily_15min_vectors.csv")
zhluk_number = pd.read_csv("daily_clusters_6.csv")

#dátum ako datetime.date
for df in [vektory, zhluk_number]:
    df["day"] = pd.to_datetime(df["day"]).dt.date

#indexovanie a zoradenie
vector_cols = list(map(str, range(96)))
vektory.set_index(["dataid", "day"], inplace=True)
vektory.sort_index(inplace=True)
zhluk_number.set_index(["dataid", "day"], inplace=True)

#spojenie zhlukov s vektormi
df_merged = vektory.join(zhluk_number, how="inner")
df_merged.reset_index(inplace=True)
df_merged[vector_cols] = df_merged[vector_cols].astype(float)
df_merged.columns = [int(col) if col in vector_cols else col for col in df_merged.columns]

#vypočet priemerných profilov zhlukov
df_merged.set_index(["dataid", "day"], inplace=True)
denne_profiles = df_merged.groupby("cluster")[list(range(96))].mean().sort_index()

print("\npriemerne profily zhlukov:")
print(denne_profiles.head())

#vypočet DTW matice na základe vzdialenosti medzi dňami
k_zhluk = denne_profiles.shape[0]
distance_matrix = np.zeros((k_zhluk, k_zhluk))

for i in range(k_zhluk):
    for j in range(k_zhluk):
        a = denne_profiles.iloc[i].values
        b = denne_profiles.iloc[j].values
        distance_matrix[i, j] = dtw(a, b)

plt.figure(figsize=(8, 6))
sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=denne_profiles.index,
            yticklabels=denne_profiles.index)
plt.title("Matica podobnosti denných zhlukov")
plt.xlabel("zhluk")
plt.ylabel("zhluk")
plt.tight_layout()
plt.show()

#ulozenie matice
df_distance = pd.DataFrame(distance_matrix, index=[f"cluster_{i}" for i in range(k_zhluk)], columns=[f"cluster_{i}" for i in range(k_zhluk)])
df_distance.to_csv("cluster_similarity_matrix.csv", index=True, float_format="%.4f")




#2)týždenne patterny 0000000 a t d
import csv

#načitanie denných zhlukov 
df = pd.read_csv("daily_clusters_6.csv")
df["day"] = pd.to_datetime(df["day"])

#pridanie tyždna
df["week"] = df["day"].dt.to_period("W")
df["week_start"] = df["week"].apply(lambda x: x.start_time)
df["week_end"] = df["week"].apply(lambda x: x.end_time)
max_date = df["day"].max()
df["week_end"] = df["week_end"].apply(lambda x: min(x, max_date))

#tvorba týždenných patternov
df_weekly = (
    df.sort_values("day")
      .groupby(["dataid", "week", "week_start", "week_end"])["cluster"]
      .apply(lambda x: "".join(map(str, x)))
      .reset_index()
)

#kontrola dlzky
df_weekly["length"] = df_weekly["cluster"].str.len()
num_total = len(df_weekly)
num_incomplete = (df_weekly["length"] < 7).sum()
num_unique = df_weekly["cluster"].nunique()

print(f"\npočet týždnov: {num_total}")
print(f"neúplné týždné (<7 dni): {num_incomplete}")
print(f"unikatné patterny: {num_unique}")
print(df_weekly["cluster"].value_counts().head(10))

#uloženie
output_path = "weekly_patterns.csv"
df_weekly["cluster"] = df_weekly["cluster"].astype(str)
df_weekly.to_csv(output_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)





#3)zhlukovanie týždenných paternov
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

#načitanie týždenných patternov (vzorov)
df_tyz_pat = pd.read_csv("weekly_patterns.csv", dtype={"cluster": str})
df_tyz_pat = df_tyz_pat[df_tyz_pat["cluster"].str.len() == 7].reset_index(drop=True)

matica_tyz = pd.read_csv("cluster_similarity_matrix.csv", index_col=0) #matica podobnosti medzi dennymi zhlukmi
vzdial_zhluk = matica_tyz.values

#funkcia pre výpočet vzdialenosti dvoch tźzdnovych vzorov
def pattern_vzdial(w1, w2, distance_matrix):
    return sum(distance_matrix[int(a)][int(b)] for a, b in zip(w1, w2))

#vypočet matice vzdialenosti medzi tyžd. patternami
n = len(df_tyz_pat)
week_dist = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        d = pattern_vzdial(df_tyz_pat.loc[i, "cluster"],
                                  df_tyz_pat.loc[j, "cluster"],
                                  vzdial_zhluk)
        week_dist[i, j] = d
        week_dist[j, i] = d

#rozsahu zhlukov a metrík
zluk_range = range(2, 17)
metrics = {"Silhouette": []}

#priemerná spotreba pre denné zhluky (na odhad tyždennej spotreby a zoradenia podľa spotreby)
priemer_daily = np.array([0.6, 1.2, 3.2, 5.5, 7.0, 9.0])

for k in zluk_range:
    #zhlukovanie
    print(f"zhlukovanie s {k} zhlukmi")
    model = KMedoids(n_clusters=k, metric="precomputed", init="k-medoids++", random_state=857)
    labels_raw = model.fit_predict(week_dist)

    #silhouette score
    sil = silhouette_score(week_dist, labels_raw, metric="precomputed")
    metrics["Silhouette"].append(sil)
    print(f"Silhouette score: {sil:.4f}")

    #odhad týždennej spotreby a zoradenie zhlukov
    df_tyz_pat["weekly_cluster_raw"] = labels_raw
    consumptions = []
    for indx in range(k):
        patterns = df_tyz_pat.loc[df_tyz_pat["weekly_cluster_raw"] == indx, "cluster"]
        totals = [sum(priemer_daily[int(ch)] for ch in pat) for pat in patterns]
        consumptions.append(np.mean(totals) if totals else 0)
    sorted_ids = np.argsort(consumptions)
    mapping = {old: new for new, old in enumerate(sorted_ids)}
    labels_sorted = np.array([mapping[l] for l in labels_raw])

    #výpis rozdelenia týždňov po triedeni
    print("rozdelenie týždňov po zhlukoch:")
    for indx, cnt in pd.Series(labels_sorted).value_counts().sort_index().items():
        print(f" ZHLUK {indx}: {cnt} týždňov")

    #ulozenie pre 5 zhlukov
    if k == 5:
        out_cols = ["dataid", "week_start", "week_end", "cluster", "weekly_cluster"]
        out_df = df_tyz_pat.copy()
        out_df["weekly_cluster"] = labels_sorted
        out_df[out_cols].to_csv(
            "weekly_patterns_clustered_5.csv",
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL
        )
        

#vykresľovanie silhouette score pre všetky k
plt.figure(figsize=(10, 4))
plt.plot(list(zluk_range), metrics["Silhouette"], marker='o', color='purple')
plt.title("Silhouette score pre tyždenné zhluky")
plt.xlabel("počet klastrov")
plt.ylabel("silhouette score")
plt.grid(True)
plt.tight_layout()
plt.show()


#4)tabulky top týžd. patternov v každom zhluku
#načitanie dat so spravným typom
df = pd.read_csv("weekly_patterns_clustered_5.csv", dtype={"cluster": "string"}, quotechar='"')

#iba tie s dlzkou 7
df = df[df["cluster"].str.len() == 7].reset_index(drop=True)

top = 5
for cluster_id in sorted(df["weekly_cluster"].unique()):
    print(f"\nZHLUK {cluster_id} – top {top} šablony:")
    top_patterns = (
        df[df["weekly_cluster"] == cluster_id]["cluster"]
        .value_counts()
        .head(top)
    )
    for pattern, count in top_patterns.items():
        print(f"  {pattern} ({count}x)")
