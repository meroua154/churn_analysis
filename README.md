# 📊 Telco Customer Churn Prediction

> Projet Data Science end-to-end : identifier les clients susceptibles de quitter un opérateur télécom et proposer des actions marketing ciblées.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Status](https://img.shields.io/badge/Status-Complet-success)

---

## 🎯 Objectif

Un opérateur télécom constate que **26.5% de ses clients** quittent le service chaque année.  
L'objectif est de :
1. **Identifier** les clients à risque avant qu'ils partent
2. **Comprendre** les facteurs qui influencent le churn
3. **Proposer** des actions marketing concrètes par segment de risque

---

## 📁 Structure du projet

```
telco-churn-prediction/
│
├── 📓 notebooks/
│   └── churn_analysis.ipynb         # Notebook principal
│
├── 📁 data/
│   └── data.csv                     # Dataset IBM Telco
│
├── 📁 outputs/
│   └── plan_marketing_segments.csv  # Clients classés par risque
│
├── 📄 README.md
└── 📄 requirements.txt
```

---

## 📦 Dataset

| Info | Détail |
|------|--------|
| Source | [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Lignes | 7 043 clients |
| Colonnes | 21 variables |
| Variable cible | `Churn` (Yes / No) |
| Déséquilibre | 73.5% No — 26.5% Yes |

---

## 🔄 Pipeline complet

### Étape 1 — Chargement & exploration
- Lecture du fichier CSV avec Pandas
- Vérification des dimensions, types et valeurs manquantes
- Détection de **11 valeurs cachées** dans `TotalCharges` (espaces vides)

### Étape 2 — Nettoyage des données
- Correction du type de `TotalCharges` : `object` → `float64`
- Remplacement des valeurs `"No internet service"` et `"No phone service"` par `"No"`
- `SeniorCitizen` déjà encodé en `0/1` — conservé tel quel
- Suppression de `TotalCharges` (corrélation 0.83 avec `tenure` — redondante)
- Suppression de `customerID` (identifiant inutile pour la modélisation)

### Étape 3 — Analyse exploratoire (EDA)

Principaux insights identifiés :

| Facteur | Taux de Churn | Niveau de risque |
|---------|--------------|-----------------|
| Paiement Electronic check | **45%** | 🔴 Critique |
| Contrat Month-to-month | **42%** | 🔴 Critique |
| Internet Fiber optic | **41%** | 🔴 Critique |
| Senior Citizen | **41%** | 🟠 Élevé |
| Sans TechSupport | **31%** | 🟠 Élevé |
| Sans OnlineSecurity | **31%** | 🟠 Élevé |

> **Insight clé** : Les clients qui partent le font dans les **13 premiers mois**.  
> Tenure court + facture élevée = signal le plus fort de churn.

### Étape 4 — Encodage & normalisation
- **Label Encoding** : colonnes binaires Yes/No → 0/1
- **One-Hot Encoding** : `InternetService`, `Contract`, `PaymentMethod` (`drop_first=True`)
- **StandardScaler** : `tenure` et `MonthlyCharges` normalisés (moyenne=0, std=1)
- Dataset final : **7 043 lignes × 22 colonnes**, 0 NaN

### Étape 5 — Modélisation

3 modèles entraînés et comparés (split 80/20, `stratify=y`) :

| Modèle | AUC-ROC | AUC-ROC (CV 5-fold) | Recall Churn | Precision Churn |
|--------|---------|---------------------|--------------|-----------------|
| **Régression Logistique (seuil 0.45)** | **0.839** | **0.843 ±0.014** | **82%** | 48% |
| XGBoost | 0.841 | — | 80% | 51% |
| Random Forest | 0.820 | 0.815 ±0.015 | 47% | 63% |

> ✅ **Modèle retenu : Régression Logistique avec seuil optimisé à 0.45**  
> Meilleur recall (82%) = 307 churners détectés sur 374.  
> Les relations étant quasi-linéaires, la LR est aussi performante que XGBoost — avec l'avantage d'être plus simple et interprétable.

### Étape 6 — Optimisation du seuil

Le seuil par défaut (0.50) a été abaissé à **0.45** pour maximiser la détection des churners :

| Seuil | Recall | Precision | Churners détectés |
|-------|--------|-----------|-------------------|
| 0.50 | 78% | 51% | 292 / 374 |
| **0.45** ✅ | **82%** | **48%** | **307 / 374** |
| 0.30 | 93% | 43% | 348 / 374 |

> Abaisser le seuil = détecter plus de churners au prix de quelques faux positifs — acceptable métier.

### Étape 7 — Segmentation marketing

4 segments définis selon la probabilité de churn prédite :

| Segment | Probabilité | Clients | Tenure moyen | Facture moyenne | Action recommandée |
|---------|------------|---------|-------------|-----------------|-------------------|
| 🔴 Critique | > 70% | **1 756** (24.9%) | 13 mois | 81$/mois | Appel + offre contrat annuel -20% |
| 🟠 Élevé | 50-70% | 1 097 (15.6%) | 24 mois | 69$/mois | Email urgent + 1 mois offert |
| 🔵 Modéré | 30-50% | 1 192 (16.9%) | 27 mois | 59$/mois | Campagne fidélité automatisée |
| 🟢 Faible | < 30% | 2 998 (42.6%) | 49 mois | 56$/mois | Parrainage + upsell |

**Profil type du client critique :**
- Contrat **Month-to-month**
- Internet **Fiber optic**
- Tenure **< 4 mois**
- Facture **> 93$/mois**
- Paiement **Electronic check**
- Probabilité de churn **> 94%**

---

## 🚀 Lancer le projet

### 1. Cloner le repo
```bash
git clone https://github.com/meroua154/churn_analysis.git
cd churn_analysis
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer Jupyter Lab
```bash
jupyter lab
```

### 4. Ouvrir le notebook
```
notebooks/churn_analysis.ipynb
```

---

## 🛠️ Stack technique

| Outil | Usage |
|-------|-------|
| `Python 3.10` | Langage principal |
| `Pandas` | Manipulation des données |
| `NumPy` | Calculs numériques |
| `Scikit-learn` | Modélisation & évaluation |
| `XGBoost` | Modèle gradient boosting |
| `Matplotlib` | Visualisations |
| `Seaborn` | Visualisations statistiques |
| `Jupyter Lab` | Environnement de développement |

---

## 📄 requirements.txt

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
jupyterlab
```

---

## 👤 Auteur

**Meroua Rezig**  
📧 merouarezig15@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/meroua-rezig-967aa9245/)  
🐙 [GitHub](https://github.com/meroua154)

---


*Dataset source : [IBM Sample Data Sets](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)*