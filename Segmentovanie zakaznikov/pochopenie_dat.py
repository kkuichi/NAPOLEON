import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#funkcia na popis dát
def pochop_data(file_path):
    print(f"\nnázov datasetu: {file_path}")

    df = pd.read_csv(file_path)

    #konverzia času
    df["local_15min"] = pd.to_datetime(df["local_15min"], errors="coerce", utc=True)
    df["local_15min"] = df["local_15min"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["day"] = df["local_15min"].dt.floor("D")


    kolko_house = df["dataid"].nunique()
    print(f"počet unikátnych domov: {kolko_house}")


    denne_profiles = df.groupby(["dataid", "day"]).size().reset_index()
    print(f"počet dennych profilov (dní spolu): {len(denne_profiles)}")


    unique_d = df["day"].nunique()
    kolko_zaznamov = len(df)
    print(f"počet unikátnych dní celkovo: {unique_d}")
    print(f"celkový počet záznamov: {kolko_zaznamov}")

    nanka = df.isna().sum().sum()
    print(f"celkový počet NaN hodnôt: {int(nanka)}")

    print("\nzoznam atribútov a ich dátové typy:")
    print(df.dtypes)

    iba_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if "dataid" in iba_numeric:
        iba_numeric.remove("dataid")
    print("\nzákladné štatistiky numerických atribútov:")
    print(df[iba_numeric].describe())

    #extrémne hodnoty
    extreme_table = df[iba_numeric].agg(['min', 'max']).T
    hard_anomalies = extreme_table[(extreme_table['min'] < -1000) | (extreme_table['max'] > 1000)]
    if not hard_anomalies.empty:
        print("\natribúty s extrémnymi hodnotami:")
        print(hard_anomalies.sort_values(by='max', ascending=False))
        count_extreme = ((df[hard_anomalies.index] < -1000) | (df[hard_anomalies.index] > 1000)).sum().sum()
        print(f"\ncelkový počet extrémnych hodnôt v df: {int(count_extreme)}")


#analýza pre všetky datasety
pochop_data("newyork.csv")
pochop_data("california.csv")
pochop_data("austin.csv")