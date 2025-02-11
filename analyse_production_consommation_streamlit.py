import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import chardet
import csv

# Fonction pour détecter l'encodage du fichier
def detect_encoding(file):
    raw_data = file.read(10000)  # Lire uniquement les premiers 10Ko pour la détection
    result = chardet.detect(raw_data)
    file.seek(0)  # Réinitialiser le pointeur de fichier
    return result['encoding']

# Fonction pour détecter le séparateur du fichier CSV
def detect_separator(file, encoding):
    sample = file.read(1000).decode(encoding)  # Lire un échantillon de 1000 octets
    file.seek(0)
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(sample).delimiter
    return delimiter

# Fonction pour détecter le nombre de lignes à ignorer (skiprows)
def detect_skiprows(file, encoding, separator):
    file.seek(0)
    for i, line in enumerate(file):
        decoded_line = line.decode(encoding)
        if decoded_line.strip() and len(decoded_line.split(separator)) > 1:
            file.seek(0)
            return i

# Fonction pour charger un fichier CSV avec détection automatique du séparateur et du skiprows
def load_csv(file):
    encoding = detect_encoding(file)
    separator = detect_separator(file, encoding)
    skiprows = detect_skiprows(file, encoding, separator)
    file.seek(0)
    return pd.read_csv(file, sep=separator, encoding=encoding, skiprows=skiprows)

# Fonction pour parser les dates avec différentes approches
def parse_date(date_str, year):
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except ValueError:
        try:
            return pd.to_datetime(date_str + f' {year}', format='%d/%m %H:%M %Y', dayfirst=True)
        except ValueError:
            return date_str

# Fonction pour harmoniser l'année des dates
def harmonize_year(date, target_year):
    if pd.isna(date):
        return date
    return date.replace(year=target_year)

# Fonction pour charger et nettoyer le fichier de production
def load_production_csv(file, study_year):
    encoding = detect_encoding(file)
    separator = detect_separator(file, encoding)
    skiprows = detect_skiprows(file, encoding, separator)
    file.seek(0)
    df = pd.read_csv(file, sep=separator, decimal='.', skiprows=skiprows, usecols=[0, 1])
    df.columns = ['time', 'P']
    df = df[df['time'].str.match(r'\d{8}:\d{4}') == True]
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M', errors='coerce')
    df['P'] = pd.to_numeric(df['P'], errors='coerce')  # Conversion explicite en numérique
    return df

# Fonction pour calculer les valeurs mensuelles et annuelles
def calculate_monthly_and_annual(df, column_name):
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, errors='coerce')
   
    df['Month'] = df.index.month
    monthly_values = df.groupby('Month')[column_name].sum() / 1000  # Conversion en kWh
    total_value = monthly_values.sum()
    return monthly_values, total_value

st.title("Analyse de Production et Consommation")

uploaded_production_file = st.file_uploader("Charger le fichier de production", type=["csv"])
uploaded_consumption_files = st.file_uploader("Charger les fichiers de consommation", type=["csv"], accept_multiple_files=True)

if uploaded_production_file and uploaded_consumption_files:
    consumption_data = []
    all_dates = []
    for uploaded_file in uploaded_consumption_files:
        df = load_csv(uploaded_file)
        if df.shape[1] >= 2:
            all_dates.extend(pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors='coerce').dropna().dt.year.tolist())
            column_name = uploaded_file.name.replace('.csv', '')
            df.columns = ['Horodate', column_name]
            df['Horodate'] = df['Horodate'].apply(lambda x: parse_date(x, max(all_dates, default=None)))
            df.set_index('Horodate', inplace=True)
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
            consumption_data.append(df)
        else:
            st.warning(f"Le fichier {uploaded_file.name} ne contient pas suffisamment de colonnes.")

    if all_dates:
        study_year = max(set(all_dates), key=all_dates.count)

        if uploaded_production_file:
            df_production = load_production_csv(uploaded_production_file, study_year)

        if consumption_data:
            df_consumption_agg = pd.concat(consumption_data, axis=1)
            df_consumption_agg.index = df_consumption_agg.index.to_series().apply(lambda x: parse_date(x, study_year) if isinstance(x, str) else x)
            df_consumption_agg.index = pd.to_datetime(df_consumption_agg.index, errors='coerce')
            df_consumption_agg.index = df_consumption_agg.index.to_series().apply(lambda x: harmonize_year(x, study_year))
            df_consumption_agg = df_consumption_agg.groupby(df_consumption_agg.index).sum()
            df_consumption_agg['Total'] = df_consumption_agg.sum(axis=1)

            # Calcul de la clé de répartition
            df_cle_repartition = df_consumption_agg.div(df_consumption_agg['Total'], axis=0).fillna(0)

            # Ajouter les colonnes jour, mois, heure pour la correspondance
            df_production['day'] = df_production['time'].dt.day
            df_production['month'] = df_production['time'].dt.month
            df_production['hour'] = df_production['time'].dt.hour

            df_cle_repartition['day'] = df_cle_repartition.index.day
            df_cle_repartition['month'] = df_cle_repartition.index.month
            df_cle_repartition['hour'] = df_cle_repartition.index.hour

            # Fusionner les DataFrames sur jour, mois, heure
            df_merged = pd.merge(df_cle_repartition, df_production, how='left', on=['day', 'month', 'hour'])

            # Effectuer la multiplication
            for col in df_cle_repartition.columns:
                if col not in ['day', 'month', 'hour']:
                    df_merged[col] = df_merged[col] * df_merged['P']

            # Ajouter l'horodate à la matrice de répartition solaire
            df_result = df_merged.drop(columns=['day', 'month', 'hour', 'P', 'time'])
            df_result.index = df_cle_repartition.index

            # Calcul du surplus
            df_surplus = df_consumption_agg.copy()
            for col in df_cle_repartition.columns:
                if col not in ['day', 'month', 'hour']:
                    df_surplus[col] = (df_surplus[col] - df_result[col]).apply(lambda x: abs(min(0, x)))

            # Calcul de la partie autoproduite
            df_autoproduite = df_result.copy()

            # Calcul de la partie alloproduite
            df_alloproduite = df_consumption_agg.copy()
            for col in df_autoproduite.columns:
                df_alloproduite[col] = (df_alloproduite[col] - df_autoproduite[col]).apply(lambda x: max(0, x))

            # Calcul des valeurs mensuelles et annuelles
            df_production['Month'] = df_production['time'].dt.month
            df_production['P_kWh'] = df_production['P'] / 1000
            df_consumption_agg['Month'] = df_consumption_agg.index.month
            df_consumption_agg['Valeur_kWh'] = df_consumption_agg['Total'] / 1000  # Conversion en kWh
            monthly_consumption = [0] * 12
            monthly_production = [0] * 12
            monthly_surplus, total_surplus = calculate_monthly_and_annual(df_surplus, 'Total')
            monthly_autoproduite = [0] * 12
            monthly_alloproduite, total_alloproduite = calculate_monthly_and_annual(df_alloproduite, 'Total')

            for month in range(1, 13):
                production_month = df_production[df_production['Month'] == month]['P_kWh'].sum()
                consommation_month = df_consumption_agg[df_consumption_agg['Month'] == month]['Valeur_kWh'].sum()
                surplus_month = monthly_surplus[month] if month in monthly_surplus.index else 0
                monthly_consumption[month - 1] = consommation_month
                monthly_production[month - 1] = production_month
                monthly_autoproduite[month - 1] = production_month - surplus_month

            total_consumption = sum(monthly_consumption)
            total_production = sum(monthly_production)
            total_autoproduite = sum(monthly_autoproduite)

            # Calcul des taux d'autoconsommation et d'autoproduction
            monthly_autoconsommation = [(autoproduit / production * 100) if production > 0 else 0 for autoproduit, production in zip(monthly_autoproduite, monthly_production)]
            monthly_autoproduction = [(autoproduit / consommation * 100) if consommation > 0 else 0 for autoproduit, consommation in zip(monthly_autoproduite, monthly_consumption)]

            total_autoconsommation = (total_autoproduite / total_production * 100) if total_production > 0 else 0
            total_autoproduction = (total_autoproduite / total_consumption * 100) if total_consumption > 0 else 0

            # Création du DataFrame pour les résultats mensuels et annuels
            monthly_data = pd.DataFrame({
                'Mois': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'Consommation (kWh)': monthly_consumption,
                'Production (kWh)': monthly_production,
                'Surplus (kWh)': monthly_surplus.values,
                'Autoproduite (kWh)': monthly_autoproduite,
                'Alloproduite (kWh)': monthly_alloproduite.values,
                'Taux d\'autoconsommation (%)': monthly_autoconsommation,
                'Taux d\'autoproduction (%)': monthly_autoproduction
            })

            totals_data = pd.DataFrame({
                'Mois': ['Total Annuel'],
                'Consommation (kWh)': [total_consumption],
                'Production (kWh)': [total_production],
                'Surplus (kWh)': [total_surplus],
                'Autoproduite (kWh)': [total_autoproduite],
                'Alloproduite (kWh)': [total_alloproduite],
                'Taux d\'autoconsommation (%)': [total_autoconsommation],
                'Taux d\'autoproduction (%)': [total_autoproduction]
            })

            result_data = pd.concat([monthly_data, totals_data], ignore_index=True)
            transposed_result_data = result_data.set_index('Mois').T

            # Afficher le tableau des résultats
            st.subheader("Tableau des Résultats Mensuels et Annuels")
            st.dataframe(transposed_result_data)

            # Ajouter les graphiques
            st.subheader("Graphiques de Production et Consommation")

            # Graphique combiné avec barres séparées
            fig, ax1 = plt.subplots(figsize=(14, 7))

            bar_width = 0.15
            months = result_data['Mois'][:-1]  # Exclure 'Total Annuel' pour les mois
            bar_positions = np.arange(len(months))

            ax1.bar(bar_positions - 2*bar_width, monthly_data['Consommation (kWh)'], width=bar_width, label='Consommation (kWh)')
            ax1.bar(bar_positions - bar_width, monthly_data['Production (kWh)'], width=bar_width, label='Production (kWh)')
            ax1.bar(bar_positions, monthly_data['Autoproduite (kWh)'], width=bar_width, label='Autoproduite (kWh)')
            ax1.bar(bar_positions + bar_width, monthly_data['Alloproduite (kWh)'], width=bar_width, label='Alloproduite (kWh)')
            ax1.bar(bar_positions + 2*bar_width, monthly_data['Surplus (kWh)'], width=bar_width, label='Surplus (kWh)')

            ax1.set_xlabel('Mois')
            ax1.set_ylabel('Énergie (kWh)')
            ax1.set_title('Consommation, Production, Autoproduite, Alloproduite et Surplus par Mois')
            ax1.set_xticks(bar_positions)
            ax1.set_xticklabels(months)
            ax1.legend(loc='upper left')

            # Ajouter un deuxième axe y pour les taux d'autoconsommation et d'autoproduction
            ax2 = ax1.twinx()
            ax2.plot(bar_positions, monthly_data['Taux d\'autoconsommation (%)'], color='r', marker='o', linestyle='-', label='Taux d\'autoconsommation (%)')
            ax2.plot(bar_positions, monthly_data['Taux d\'autoproduction (%)'], color='b', marker='o', linestyle='-', label='Taux d\'autoproduction (%)')
            ax2.set_ylabel('Pourcentage (%)')
            ax2.legend(loc='upper right')

            st.pyplot(fig)

            # Boutons de téléchargement des matrices
            st.download_button("Télécharger la matrice de données de consommation en CSV", df_consumption_agg.to_csv().encode('utf-8'), file_name='consommation_aggregate.csv', mime='text/csv')
            st.download_button("Télécharger la matrice des clés de répartition en CSV", df_cle_repartition.to_csv().encode('utf-8'), file_name='cle_repartition.csv', mime='text/csv')
            st.download_button("Télécharger la matrice de répartition solaire en CSV", df_result.to_csv().encode('utf-8'), file_name='repartition_solaire.csv', mime='text/csv')
            st.download_button("Télécharger la matrice de surplus en CSV", df_surplus.to_csv().encode('utf-8'), file_name='surplus.csv', mime='text/csv')
            st.download_button("Télécharger la matrice de la partie autoproduite en CSV", df_autoproduite.to_csv().encode('utf-8'), file_name='autoproduite.csv', mime='text/csv')
            st.download_button("Télécharger la matrice de la partie alloproduite en CSV", df_alloproduite.to_csv().encode('utf-8'), file_name='alloproduite.csv', mime='text/csv')
            st.download_button("Télécharger le tableau des résultats en CSV", transposed_result_data.to_csv().encode('utf-8'), file_name='resultats.csv', mime='text/csv')
            
