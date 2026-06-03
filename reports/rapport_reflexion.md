# Rapport court - Agriculture au Burundi

## Questions 29 et 30

### Q29 - Scenario Gitega - Haricot

Le scenario Gitega - Haricot combine une pluviometrie tres faible (430 mm), pas d'engrais et pas d'irrigation. Les predictions obtenues sont les suivantes :

| Modele | Prediction | Probabilite bonne |
|---|---:|---:|
| Arbre de decision | Bonne | 97.9% |
| Foret aleatoire | Bonne | 81.0% |
| Regression logistique | Bonne | 91.1% |

Cette situation est agronomiquement risquee : le haricot est sensible au stress hydrique, surtout lorsque la pluie est faible et que l'irrigation n'est pas disponible. Un agronome pourrait recommander une meilleure conservation de l'humidite du sol, l'acces a une source d'irrigation si possible, l'ajustement du calendrier cultural, l'utilisation de varietes plus tolerantes et un appui en fertilisation.

### Q30 - Choix du modele pour le Ministere

Le modele recommande est la Regression logistique si l'objectif est la robustesse globale, car elle obtient l'AUC la plus elevee (0.839). En accuracy pure, le meilleur modele est Regression logistique (0.940).

Pour une utilisation publique, la foret aleatoire est souvent le meilleur compromis : elle gere bien les interactions non lineaires entre pluie, altitude, temperature, culture et pratiques agricoles, tout en restant interpretable avec l'importance des variables.

Des donnees supplementaires amelioreraient le modele : type de sol, dates exactes de semis, maladies et ravageurs, qualite des semences, prix/acces aux intrants, donnees meteo plus fines, pratiques culturales et historiques de rendement par exploitation.

Limites : le dataset est simule, la cible est construite a partir du rendement, les classes sont desequilibrees, certaines valeurs sont manquantes et les scenarios incomplets utilisent des valeurs par defaut. Le modele doit donc aider la decision, pas remplacer l'expertise agronomique.

## Resume technique

- Lignes/colonnes : 1620 / 14
- Periode : 2015-2023
- Provinces : 15
- Cultures : Bananier, Haricot, Manioc, Maïs, Patate douce, Sorgho
- Meilleure culture en rendement moyen : Manioc
- Province avec pluie moyenne la plus forte : Muramvya
- Profondeur avec meilleure accuracy test : 1