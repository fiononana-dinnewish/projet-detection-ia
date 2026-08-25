# Comparaison learning rates (15 epochs, data augmentation)

| lr     | best_val_accuracy | stabilité |
|--------|-------------------|-----------|
| 0.0001 | 91.95%             | stable, progresse encore à ep15 |
| 0.001  | 99.33%             | instable (chute à 93% ensuite) |
| 0.0005 | 98.66%             | assez instable |

Prochaine étape : tester lr=0.0001 avec 25 epochs, puis évaluation sur test set.