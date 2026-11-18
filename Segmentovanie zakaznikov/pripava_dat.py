import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#nastavenia zobrazenia
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.set_option('float_format', '{:f}'.format)

#prvá časť
#čistenie a interpolácia extrémnych hodnôt

def cistime(file_path):
    print(f"\ndataset: {file_path}")
    df = pd.read_csv(file_path)
    df["local_15min"] = pd.to_datetime(df["local_15min"], errors="coerce", utc=True)

    iba_numeric = df.select_dtypes(include=[np.number]).columns

    sns.boxplot(data=df[iba_numeric])
    plt.xticks(rotation=90)
    plt.title("distribúcia dát pred čistením")
    plt.show()

    #interpolácia mimo ±3 odchýlky
    for col in iba_numeric:
        mean = df[col].mean()
        std = df[col].std()
        minusove = mean - 3 * std #horná hranica
        plusove = mean + 3 * std  #dolná hranica
        if ((df[col] < minusove) | (df[col] > plusove)).sum() > 0:
            orig_nan = df[col].isna()  #NaN čo boli pred tým ponecháme
            df[col] = df[col].mask((df[col] < minusove) | (df[col] > plusove)).interpolate()
            df.loc[orig_nan, col] = np.nan


    sns.boxplot(data=df[iba_numeric])
    plt.xticks(rotation=90)
    plt.title("distribúcia dát po čistení")
    plt.show()

    return df

#druhá časť
#agregácia atributov podľa kategórie
def aggregate_kategoria(df, label):
    print(f"\nagregujem dataset: {label}")
    df = df.copy()
    df["local_15min"] = pd.to_datetime(df["local_15min"], errors="coerce", utc=True)
    df["hour"] = df["local_15min"].dt.hour
    df["day"] = df["local_15min"].dt.date

    mapping = {
        "climate_energy": [
            'air1', 'air2', 'air3', 'airwindowunit1',
            'heater1', 'heater2', 'heater3',
            'furnace1', 'furnace2', 'housefan1', 'venthood1'
        ],
        "kitchen_energy": [
            'kitchen1', 'kitchen2', 'kitchenapp1', 'kitchenapp2',
            'microwave1', 'oven1', 'oven2', 'range1',
            'refrigerator1', 'refrigerator2', 'freezer1',
            'winecooler1', 'dishwasher1', 'disposal1', 'icemaker1'
        ],
        "lighting_energy": [
            'lights_plugs1', 'lights_plugs2', 'lights_plugs3',
            'lights_plugs4', 'lights_plugs5', 'lights_plugs6',
            'outsidelights_plugs1', 'outsidelights_plugs2'
        ],
        "laundry_energy": [
            'clotheswasher1', 'clotheswasher_dryg1', 'drye1', 'dryg1'
        ],
        "water_energy": [
            'pump1', 'wellpump1', 'sumppump1', 'sewerpump1',
            'sprinkler1', 'waterheater1', 'waterheater2',
            'jacuzzi1', 'pool1', 'pool2', 'poollight1',
            'poolpump1', 'aquarium1', 'circpump1'
        ],
        "rooms_energy": [
            'bedroom1', 'bedroom2', 'bedroom3', 'bedroom4', 'bedroom5',
            'livingroom1', 'livingroom2', 'diningroom1', 'diningroom2',
            'office1', 'bathroom1', 'bathroom2', 'garage1', 'garage2',
            'shed1', 'utilityroom1', 'security1', 'car1', 'car2'
        ],
        "solar_energy": ['solar', 'solar2'],
        "line_voltage": ['leg1v', 'leg2v'],
        "battery_energy": ['battery1'],
        "grid_energy": ['grid']
    }

    cela_spotreba = [
        "climate_energy", "kitchen_energy", "lighting_energy",
        "laundry_energy", "water_energy", "rooms_energy"
    ]
    total_cols = sum([mapping[k] for k in cela_spotreba], [])

    #minusove hodnoty na 0
    for col in total_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    #spočítanie celkovej spotreby a odstránenie nepotrebných cols
    for new_col, cols in mapping.items():
        df[new_col] = df[cols].sum(axis=1).round(4).astype(np.float32)
        df.drop(columns=cols, inplace=True, errors='ignore')


    df["total_consumption"] = df[cela_spotreba].sum(axis=1)
    df = df.round(4)

    print("\nstatistika po agregacii:")
    print(df.describe())
    print("\nchybajuce hodnoty po:")
    print(df.isnull().sum())

    return df

#tretia časť
#spustenie funkcie pre čistenie
ny_df = cistime("newyork.csv")
ca_df = cistime("california.csv")
tx_df = cistime("austin.csv")

#spustenie agregacie
ny_df = aggregate_kategoria(ny_df, "New York")
ca_df = aggregate_kategoria(ca_df, "California")
tx_df = aggregate_kategoria(tx_df, "Texas")

#prevod časových pásiem do ich lokálneho času a pridanie stĺpca s nazvou štátu do datasetov
ny_df["local_15min"] = ny_df["local_15min"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
ca_df["local_15min"] = ca_df["local_15min"].dt.tz_convert("US/Pacific").dt.tz_localize(None)
tx_df["local_15min"] = tx_df["local_15min"].dt.tz_convert("US/Central").dt.tz_localize(None)
ny_df["state"] = "New York"
ca_df["state"] = "California"
tx_df["state"] = "Texas"

#merge datasetov do jedneho spoločneho
boss_df = pd.concat([ny_df, ca_df, tx_df], ignore_index=True)
boss_df["hour"] = boss_df["local_15min"].dt.hour
boss_df["minute"] = boss_df["local_15min"].dt.minute
boss_df["day"] = boss_df["local_15min"].dt.date
boss_df["total_consumption"] = boss_df[["climate_energy","kitchen_energy","lighting_energy","laundry_energy","water_energy","rooms_energy"]].sum(axis=1).fillna(0)


#boss_df.to_csv("merged_df.csv", index=False, float_format="%.4f")

#tvorba 15-minútových vektorov za každý deň
boss_df["step_15min"] = ((boss_df["hour"] * 60 + boss_df["minute"]) // 15)
vec_df = (boss_df.groupby(["dataid","day","step_15min"])["total_consumption"].mean().unstack(fill_value=pd.NA).reindex(columns=range(96))) #priemer ak sa vyskytnú duplicitné časy

#rozdelenie na úplné a neúplné dni, výber len úplných
absencie = vec_df.isna().sum(axis=1)
uplne_days= vec_df[absencie == 0]
neuplne_days = vec_df[absencie > 0]

uplne_days.to_csv("daily_15min_vectors.csv", float_format="%.4f")


print(f"\ncelodenné vektory: {len(uplne_days)}")
print(f"počet neúplných dni: {len(neuplne_days)}")
if not neuplne_days.empty:
    print("\nrozdelenie absencie v neúplných dňoch:")
    print(absencie.value_counts().sort_index())

all_dates = pd.date_range(boss_df["day"].min(), boss_df["day"].max())
unique_dates = pd.to_datetime(boss_df["day"].unique())
missing_dates = all_dates.difference(unique_dates)
print(f"\nchýbajúce kalendárné dni: {len(missing_dates)}")

print("\nprvé vektory pre DTW:")
print(uplne_days.head())

