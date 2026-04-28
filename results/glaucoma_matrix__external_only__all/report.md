# Rapport d'evaluation RETFound glaucome

Portee: **comparaison externe uniquement**.

Jeu de donnees evalue: **train + val + test fusionnes**.

Chaque ligne correspond a un checkpoint fine-tune sur le dataset source, puis evalue sur le dataset cible.

| Source checkpoint | Dataset cible | N images | Accuracy | AUROC | F1 macro | Kappa | Composite |
|---|---|---:|---:|---:|---:|---:|---:|
| Glaucoma_fundus | PAPILA | 488 | 0.3176 | 0.5717 | 0.2277 | 0.0623 | 0.2872 |
| PAPILA | Glaucoma_fundus | 1544 | 0.5168 | 0.6838 | 0.2506 | 0.0284 | 0.3209 |

## Meilleure validation externe

- Source: `PAPILA`; cible: `Glaucoma_fundus`; composite: `0.3209`; AUROC: `0.6838`.
