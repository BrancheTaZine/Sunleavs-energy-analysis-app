# Sunleavs Energy Analysis App

Sunleavs Energy Analysis App est une application basée sur Streamlit qui permet aux utilisateurs d'analyser les données de production et de consommation d'énergie. L'application fournit des visualisations et des métriques détaillées pour aider les utilisateurs à comprendre leurs habitudes de consommation et de production d'énergie.

## Lien vers l'application

Accédez à l'application en ligne : [Sunleavs Energy Analysis App](https://sunleavs-energy-analysis-app-tomrougerie.streamlit.app/)

## Fonctionnalités

- Chargement des fichiers CSV pour les données de production et de consommation
- Détection et analyse automatique des fichiers CSV
- Calcul des métriques mensuelles et annuelles pour la production, la consommation, le surplus et l'autoproduction d'énergie
- Visualisation des données énergétiques avec des graphiques interactifs
- Téléchargement des résultats au format CSV

## Sources des données

- **Données de production** : Les fichiers CSV de production sont récupérés depuis PVGIS (simulation).
- **Données de consommation** : Les fichiers de consommation sont obtenus depuis Enedis avec accords et le PDL.

## Utilisation

1. **Charger des fichiers CSV :**
   - Utilisez le téléchargeur de fichiers pour charger vos fichiers CSV de production et de consommation.
   - L'application détectera automatiquement le format du fichier et analysera les données.

2. **Voir l'analyse :**
   - L'application affichera les métriques mensuelles et annuelles pour la production, la consommation, le surplus et l'autoproduction d'énergie.
   - Des graphiques interactifs vous aideront à visualiser les données.

3. **Télécharger les résultats :**
   - Vous pouvez télécharger les résultats au format CSV en utilisant les boutons de téléchargement fournis.

## Installation ( uniquement si vous voulez toucher au code )

Pour exécuter l'application localement, suivez ces étapes :

1. **Clonez le dépôt :** 

    ```bash
    git clone https://github.com/BrancheTaZine/Sunleavs-energy-analysis-app.git
    cd Sunleavs-energy-analysis-app
    ```

2. **Créez un environnement virtuel (optionnel mais recommandé) :**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Sous Windows, utilisez `venv\Scripts\activate`
    ```

3. **Installez les dépendances :**

    ```bash
    pip install -r requirements.txt
    ```

4. **Exécutez l'application :**

    ```bash
    streamlit run analyse_production_consommation_streamlit.py
    ```

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Contact

Pour toute question ou demande, veuillez contacter :

- **Nom :** Votre Nom
- **Email :** your.email@example.com
