#1) matica vzdialenosti pre ročne vektory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#načítanie týždenných zhlukov
df = pd.read_csv("weekly_patterns_clustered_5.csv", dtype={"cluster": str})
df = df[df["cluster"].str.len() == 7].copy()

#matica podobnosti denných zhlukov (použita pre zhlukovanie tyžd. patternov)
daily_sim_matrix = pd.read_csv("cluster_similarity_matrix.csv", index_col=0).values

#výpočet DTW vzdialenosť medzi dvoma týždennými vzormi w1 a w2 (zložené z denných zhlukov)
#w1, w2 sú reťazce 7 znakov (zhluk každého dňa)
def vzdial_week_patt(w1, w2, dist_matrix):
    n, m = len(w1), len(w2)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a = int(w1[i - 1])
            b = int(w2[j - 1])
            cost = dist_matrix[a][b]                                            #vľavo, vpravo a po diagonále
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]) #výber minimálnej cesty z troch možností
    return dp[n, m] #DTW vzdialenosť za celý týždeň

#výpočet medoidov pre každý týždenný zhluk
week_zhluky = sorted(df["weekly_cluster"].unique())
medoids = {}

for cluster_id in week_zhluky:
    patterns = df[df["weekly_cluster"] == cluster_id]["cluster"].tolist()
    n = len(patterns)
    vzdial_matica = np.zeros((n, n)) #matica vzdialeností medzi všetkými dvojicami vzorov
    for i in range(n):
        for j in range(i + 1, n):
            dist = vzdial_week_patt(patterns[i], patterns[j], daily_sim_matrix)
            vzdial_matica[i, j] = dist
            vzdial_matica[j, i] = dist
    medoid_index = np.argmin(vzdial_matica.sum(axis=1))
    medoids[cluster_id] = patterns[medoid_index]

#matica vzdialeností medzi medoidmi - daľej bude použita na zhlukovanie ročných vektorov
r = len(medoids)
vzdial_matica = np.zeros((r, r))

for i, a in enumerate(week_zhluky):
    for j, b in enumerate(week_zhluky):
        if i <= j:
            dist = vzdial_week_patt(medoids[a], medoids[b], daily_sim_matrix)
            vzdial_matica[i, j] = dist
            vzdial_matica[j, i] = dist

#uloženie výslednej matice a vizualizácia
df_result = pd.DataFrame(
    vzdial_matica,
    index=[f"{i}" for i in week_zhluky],
    columns=[f"{i}" for i in week_zhluky]
)
df_result.to_csv("weekly_cluster_similarity_matrix.csv", float_format="%.4f")

plt.figure(figsize=(8, 6))
sns.heatmap(df_result, annot=True, fmt=".2f", cmap="PuRd")
plt.title("Vzdialenosti medzi medoidami týždenných zhlukov")
plt.xlabel("týždenný zhluk")
plt.ylabel("týždenný zhluk")
plt.tight_layout()
plt.show()


#ročné vektry
#načítanie dát
tyzd_patterns_df = pd.read_csv("weekly_patterns_clustered_5.csv", dtype={"cluster": str})
tyzd_patterns_df["week_start"] = pd.to_datetime(tyzd_patterns_df["week_start"], errors="coerce")

#odstranenie neúplných týždňov
tyzd_patterns_df = tyzd_patterns_df[tyzd_patterns_df["cluster"].str.len() == 7].copy()

#výpočet ISO roka a čísla týždňa (rok nie je viazaný na dátum, ale na poradie týždňov začínajúce pondelkom)
tyzd_patterns_df["_year"] = tyzd_patterns_df["week_start"].dt.isocalendar().year
tyzd_patterns_df["_week"] = tyzd_patterns_df["week_start"].dt.isocalendar().week

#výtvorenie ročných vektorov
vectors = []
missing_stats = []

for (dataid, year), group in tyzd_patterns_df.groupby(["dataid", "_year"]):
    if len(group) < 48:
        continue  # preskočme neúplné roky

    #Mapa: číslo týždňa -- zhluk
    week_to_cluster = dict(zip(group["_week"], group["weekly_cluster"]))

    vector = []
    for week_num in range(1, 53):  # 52 týždňov (štandard), ak chýba týždeň tak nahradime -1
        cluster = week_to_cluster.get(week_num, -1)
        vector.append(cluster)

    n_missing = vector.count(-1)
    missing_stats.append({"dataid": dataid, "year": year, "n_missing": n_missing})

    row = {"dataid": dataid, "year": year}
    for i in range(52):
        row[f"w{i}"] = vector[i]

    vectors.append(row)

#výstupné dáta
df_vectors = pd.DataFrame(vectors)
df_vectors.to_csv("yearly_weekly_vectors.csv", index=False)
print(f"počet uložených ročných vektorov: {len(df_vectors)}")
print(df_vectors.head())

#štatistika chýbajúcich týždňov
missing_df = pd.DataFrame(missing_stats)
missing_df.to_csv("missing_week_stats.csv", index=False)
print("\nŠtatistika chýbajúcich týždňov:")
print(missing_df.head())



#3)zhlukovanie ročných vektorov
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

#načitanie dat
df = pd.read_csv("yearly_weekly_vectors.csv")
weekly_sim = pd.read_csv("weekly_cluster_similarity_matrix.csv", index_col=0).values
week_cols = [f"w{i}" for i in range(52)]


df["n_missing"] = df[week_cols].apply(lambda row: (row == -1).sum(), axis=1)
print("\nrozlozenie poctu -1 v rocnych vektoroch:")
print(df["n_missing"].value_counts().sort_index())

#filtrovanie, max 3 absencie
tyzd_absen = 3
df = df[df["n_missing"] <= tyzd_absen].reset_index(drop=True)

#náhrada -1 najbližším susedom
def tyzd_absen_na_nearest_neighbor(row):
    row_hodnoty = row[week_cols].tolist() #hodnoty vektora
    for i in range(len(row_hodnoty)):
        if row_hodnoty[i] == -1:
            left = right = None
            for l in range(i - 1, -1, -1):
                if row_hodnoty[l] != -1:
                    left = row_hodnoty[l]
                    break
            for r in range(i + 1, len(row_hodnoty)):
                if row_hodnoty[r] != -1:
                    right = row_hodnoty[r]
                    break
            if left is not None and right is not None:
                row_hodnoty[i] = left if (i - l) <= (r - i) else right
            elif left is not None:
                row_hodnoty[i] = left
            elif right is not None:
                row_hodnoty[i] = right
    return pd.Series(row_hodnoty, index=week_cols)

print(f"\npočet ročnych vektorov po filtrovani: {len(df)}")
df[week_cols] = df.apply(tyzd_absen_na_nearest_neighbor, axis=1) #nahradzovanie -1 najblizsim tyzdennym zhlukom

#vypočet DTW medzi vektormi
def dtw_rok_vector(v1, v2, sim_matrix):
    n, m = len(v1), len(v2)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a = int(v1[i - 1])
            b = int(v2[j - 1])
            cost = sim_matrix[a][b]                                             #vľavo, vpravo a po diagonále
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]) #výber minimálnej cesty z troch možností
    return dp[n, m]

vectors = df[week_cols].values
n = len(vectors)
distance_matrix = np.zeros((n, n))

print("\nvypočet DTW matice medzi ročnymi vektormi...")
for i in range(n):
    for j in range(i + 1, n):
        dist = dtw_rok_vector(vectors[i], vectors[j], weekly_sim)
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist

np.save("yearly_distance_matrix.npy", distance_matrix)

#rozsah
silhouette_scores = []
zluk_range = range(2, 10)
#zhlukovanie
for k in zluk_range:
    print(f"\nzhlukovanie {k} zhlukov")
    model = KMedoids(n_clusters=k, metric="precomputed", init="k-medoids++", random_state=828)
    labels = model.fit_predict(distance_matrix)
    sil_score = silhouette_score(distance_matrix, labels, metric="precomputed")
    silhouette_scores.append(sil_score)

    print(f"Silhouette Score: {sil_score:.4f}")
    counts = pd.Series(labels).value_counts().sort_index()
    for cluster_id, count in counts.items():
        print(f"ZHLUK {cluster_id}: {count} domacnosti")

    #vypočet priemernej spotreby pre zoradenie
    cluster_avg_spotreba = []
    for indx in range(k):
        members = df[labels == indx][week_cols].values
        #odhadneme spotrebu ako priemer hodnot cez tyždne a cez dni
        avg_spotreba = members.mean()
        cluster_avg_spotreba.append(avg_spotreba)

    zoradene_ids = np.argsort(cluster_avg_spotreba)
    old_to_new = {old: new for new, old in enumerate(zoradene_ids)}
    labels_reordered = np.vectorize(old_to_new.get)(labels)

    #uloženie len pre 4 zhluky, lebo 4 otimálna hodnota
    if k == 4:
        df_out = df.copy()
        df_out["cluster"] = labels_reordered
        df_out.to_csv("yearly_clustered_k4.csv", index=False)

plt.figure(figsize=(8, 4))
plt.plot(zluk_range, silhouette_scores, marker="o", color='orange')
plt.title("Silhouette Score pre rôzne počty ročných zhlukov")
plt.xlabel("počet zhlukov")
plt.ylabel("hodnota")
plt.grid(True)
plt.tight_layout()
plt.show()



#tabuľka mesačne štatistiky a ročný priemer
import pandas as pd

df_rok = pd.read_csv("yearly_clustered_k4.csv")         #obsahuje dataid, year, cluster
df_den = pd.read_csv("daily_15min_vectors.csv")          #obsahuje dataid, day a (15-min intervaly

#agregacia na mesiace
df_den["day"] = pd.to_datetime(df_den["day"])
df_den["year"] = df_den["day"].dt.year
df_den["month"] = df_den["day"].dt.month

#výpočet dennej spotreby
_15min = [str(i) for i in range(96)]
df_den["daily_sum"] = df_den[_15min].sum(axis=1)

#spojenie s ročným zhlukom
df_rok_short = df_rok[["dataid", "year", "cluster"]]
df_spojeny = df_den.merge(df_rok_short, on=["dataid", "year"], how="inner")

#výpočet mesačnej spotreby pre každý dom
monthly_dom_sum = (df_spojeny.groupby(["cluster", "dataid", "month"])["daily_sum"].sum().reset_index())

#priemer mesačných súm v rámci každého zhluku
mesiac_avg = (monthly_dom_sum.groupby(["cluster", "month"])["daily_sum"]
    .mean()
    .reset_index()
    .pivot(index="month", columns="cluster", values="daily_sum")
    .round(2))

#priemer za celý rok
yearly_dom_sum = df_spojeny.groupby(["cluster", "dataid"])["daily_sum"].sum().reset_index()
rok_avg = (yearly_dom_sum.groupby("cluster")["daily_sum"]
    .mean()
    .round(2)
    .rename("Year_avg"))


mesiac_avg.index = mesiac_avg.index.map({
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",
    6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct",
    11: "Nov", 12: "Dec"
})
mesiac_avg.loc["Year_avg"] = rok_avg

mesiac_avg.to_csv("cluster_monthly_statistics.csv")
print(mesiac_avg)
