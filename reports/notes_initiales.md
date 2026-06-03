# Notes initiales

L'enonce demande un pipeline ML complet pour predire `bonne_recolte`.

Constats principaux :

- Dataset : 1620 lignes et 14 colonnes.
- Periode : 2015 a 2023.
- Provinces : 15.
- Cultures : 6.
- Cible tres desequilibree : la grande majorite des lignes etiquetees sont des bonnes recoltes.
- Valeurs manquantes : `pluviometrie_mm`, `utilisation_engrais`, `rendement_t_ha`, `production_totale_t`, `bonne_recolte`.
- Colonnes a exclure du modele : `rendement_t_ha` et `production_totale_t`, car elles creent un data leakage.

Le fichier `ChatBot.py` est un notebook JSON contenant un petit chatbot generaliste. Il n'est pas utile pour ce TP agricole.
