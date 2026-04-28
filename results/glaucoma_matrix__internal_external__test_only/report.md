# Rapport d'evaluation RETFound glaucome

Portee: **comparaison externe et interne**.

Jeu de donnees evalue: **split test uniquement**.

Chaque ligne correspond a un checkpoint fine-tune sur le dataset source, puis evalue sur le dataset cible.

| Source checkpoint | Dataset cible | N images | Accuracy | AUROC | F1 macro | Kappa | Composite |
|---|---|---:|---:|---:|---:|---:|---:|
| Glaucoma_fundus | Glaucoma_fundus | 465 | 0.8430 | 0.9428 | 0.7906 | 0.7421 | 0.8252 |
| Glaucoma_fundus | PAPILA | 98 | 0.2857 | 0.5787 | 0.2044 | 0.0522 | 0.2784 |
| PAPILA | Glaucoma_fundus | 465 | 0.5204 | 0.6794 | 0.2627 | 0.0423 | 0.3281 |
| PAPILA | PAPILA | 98 | 0.7755 | 0.8493 | 0.5235 | 0.4232 | 0.5987 |

## Meilleure validation externe

- Source: `PAPILA`; cible: `Glaucoma_fundus`; composite: `0.3281`; AUROC: `0.6794`.

## Meilleure validation interne

- Source et cible: `Glaucoma_fundus`; composite: `0.8252`; AUROC: `0.9428`.
