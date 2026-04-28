# RETFound - Evaluation croisee de deux modeles fine-tunes pour le glaucome

> Projet autonome de comparaison de la generalisation cross-dataset de deux checkpoints RETFound fine-tunes pour l'identification du glaucome, a partir des datasets **Glaucoma_fundus** et **PAPILA**.

## Objectif

Ce projet reprend l'organisation du dossier d'exemple `RETFound_Comparaison_For_Diabetic_Retinopathy_Evaluation`, mais l'adapte a une evaluation glaucome a **2 checkpoints x 2 datasets** :

- checkpoint fine-tune sur `Glaucoma_fundus`
- checkpoint fine-tune sur `PAPILA`
- evaluation croisee sur les deux datasets

Le modele de fondation reste **RETFound** avec une tete de classification adaptee a **3 classes**.

## Convention commune des classes

Les deux datasets n'utilisent pas exactement les memes noms de dossiers, mais ils peuvent etre alignes sur une convention clinique commune :

| Label | Convention commune | Glaucoma_fundus | PAPILA |
|---|---|---|---|
| 0 | Normal | `anormal_control` | `anormal` |
| 1 | Suspect / precoce | `bearly_glaucoma` | `bsuspectglaucoma` |
| 2 | Glaucome confirme / avance | `cadvanced_glaucoma` | `cglaucoma` |

Cette harmonisation est un choix d'evaluation necessaire pour comparer les checkpoints entre datasets differents.

## Structure du projet

```text
RETFound_Comparaison_For_Glaucoma_Evaluation/
|-- README.md
|-- requirements.txt
|-- check/
|-- dataset/
|-- results/
|-- retfound_eval/
|   |-- __init__.py
|   |-- config.py
|   |-- data.py
|   |-- device.py
|   |-- evaluate.py
|   |-- metrics.py
|   |-- model.py
|   `-- plots.py
`-- scripts/
    |-- prepare_local_assets.py
    |-- inspect_glaucoma_datasets.py
    |-- evaluate_pair.py
    |-- evaluate_matrix.py
    |-- summarize_results.py
    `-- make_markdown_report.py
```

## Prerequis

- Python 3.10 ou plus recent recommande
- acces internet pour telecharger les assets Google Drive

Installation :

```bash
python3 -m pip install -r requirements.txt
```

## Preparation des assets locaux

Le projet ne copie plus les datasets et checkpoints depuis un dossier voisin. Il telecharge maintenant directement les assets dans `dataset/` et `check/` a partir des liens Google Drive fournis.

Commande standard :

```bash
python3 scripts/prepare_local_assets.py
```

Ce script :

- telecharge et dezippe `Glaucoma_fundus` dans `dataset/Glaucoma_fundus`
- telecharge et dezippe `PAPILA` dans `dataset/PAPILA`
- telecharge le checkpoint de `Glaucoma_fundus` puis le renomme en `checkpoint-best-Glaucoma_fundus.pth`
- telecharge le checkpoint de `PAPILA` puis le renomme en `checkpoint-best-PAPILA.pth`

Options utiles :

```bash
python3 scripts/prepare_local_assets.py --force
python3 scripts/prepare_local_assets.py --skip-datasets
python3 scripts/prepare_local_assets.py --skip-checkpoints
```

## Utilisation

Placez-vous dans le dossier du projet :

```bash
cd /Users/megretleo/scr/2026/RETFound_Comparaison_For_Glaucoma_Evaluation
```

### 1. Inspecter les datasets

```bash
python3 scripts/inspect_glaucoma_datasets.py
```

### 2. Evaluer une paire source -> cible

Exemple de comparaison externe sur tous les splits fusionnes :

```bash
python3 scripts/evaluate_pair.py \
  --train-dataset Glaucoma_fundus \
  --eval-dataset PAPILA \
  --all \
  --batch-size 16
```

Exemple en test uniquement :

```bash
python3 scripts/evaluate_pair.py \
  --train-dataset PAPILA \
  --eval-dataset PAPILA \
  --test-only \
  --batch-size 16
```

### 3. Lancer la matrice complete selon les deux scenarios metier

#### Scenario A - Rapport de comparaison externe uniquement

Objectif : comparer uniquement les validations croisees entre datasets differents, en fusionnant `train + val + test`.

Commande :

```bash
python3 scripts/evaluate_matrix.py --external-only --all --batch-size 16
```

Puis regeneration du resume et du rapport :

```bash
python3 scripts/summarize_results.py --external-only --all
python3 scripts/make_markdown_report.py --external-only --all
```

Ce scenario cree par defaut :

```text
results/glaucoma_matrix__external_only__all/
```

#### Scenario B - Rapport de comparaison externe et interne

Objectif : comparer a la fois les diagonales internes et les validations externes, mais uniquement sur le split `test`.

Commande :

```bash
python3 scripts/evaluate_matrix.py --test-only --batch-size 16
```

Puis regeneration du resume et du rapport :

```bash
python3 scripts/summarize_results.py --test-only
python3 scripts/make_markdown_report.py --test-only
```

Ce scenario cree par defaut :

```text
results/glaucoma_matrix__internal_external__test_only/
```

### 4. Utiliser des chemins de sortie personnalises

Si besoin, vous pouvez toujours forcer un dossier de sortie specifique :

```bash
python3 scripts/evaluate_matrix.py --external-only --all --output-dir results/mon_run
python3 scripts/summarize_results.py --results-dir results/mon_run
python3 scripts/make_markdown_report.py --summary results/mon_run/summary_metrics.csv --output results/mon_run/report.md
```

## Sens des nouveaux modes

Le projet distingue maintenant explicitement deux usages :

- `--all` : fusionne `train`, `val` et `test` pour evaluer sur l'ensemble complet du dataset cible
- `--test-only` : n'utilise que `test`, pour une comparaison plus stricte et plus proche d'une evaluation finale

En pratique :

- `--external-only --all` correspond au rapport de comparaison externe uniquement que tu souhaites
- `--test-only` sans `--external-only` correspond au rapport de comparaison externe et interne en ne gardant que les fichiers de test

Les options `--all` et `--test-only` sont mutuellement exclusives.

## Sorties produites

Pour chaque paire evaluee, le projet cree un dossier :

```text
results/.../train-{SOURCE}__eval-{TARGET}/
```

Contenu :

- `predictions.csv` : predictions image par image
- `metrics.csv` : metriques globales
- `metrics.json` : metriques detaillees, rapport par classe, matrice de confusion
- `confusion_matrix.png` : matrice de confusion normalisee
- `roc_curves.png` : courbes ROC one-vs-rest par classe

Au niveau de la campagne :

- `summary_metrics.csv`
- `report.md`

Les metriques incluent maintenant aussi :

- `split_mode` : `all`, `test_only` ou `custom`
- `comparison_scope` : `external_only` ou `internal_external`

## Notes techniques

- Architecture : **RETFound compatible ViT-L/16**
- Taille d'entree : `224 x 224`
- Normalisation : statistiques ImageNet
- Chargement des checkpoints : prise en charge des cles `model`, `state_dict` ou dict brut
- Device `auto` :
  - macOS Apple Silicon : priorite a `mps`
  - Linux/Windows avec GPU NVIDIA : priorite a `cuda`
  - sinon : `cpu`

## Conseils selon l'environnement

### macOS

- `--device auto` choisit `mps` si disponible
- si vous constatez un souci sur MPS, utilisez `--device cpu`
- gardez `--num-workers 0` si vous voulez privilegier la stabilite

### Windows

- preferez `python scripts/...` ou `py scripts/...`
- si l'ouverture multi-processus pose probleme, utilisez `--num-workers 0`

### Sans GPU

Le projet fonctionne integralement sur CPU, avec un temps d'evaluation plus long. Aucun changement de code n'est necessaire :

```bash
python3 scripts/evaluate_matrix.py --test-only --device cpu
```

## Hypotheses importantes

- Les checkpoints locaux correspondent bien a une tete de classification a 3 classes.
- L'alignement `precoce <-> suspect` et `avance <-> glaucome` est suppose acceptable pour une comparaison cross-dataset.
- Les dossiers `train`, `val`, `test` sont attendus dans les archives telechargees.

## Reference scientifique

> Zhou, Y., et al. *A foundation model for generalizable disease detection from retinal images.* Nature, 2023. DOI: `10.1038/s41586-023-06555-x`
