# Configs simplifiées

- `config/base/` : paramètres physiques partagés (mixing, masses, résolution)
- `config/presets/` : configurations prêtes à lancer par opération

## Convention de nommage

`<type_operation>_<mode>.txt`

Exemples :

- `energy_3p1.txt`
- `energy_3p2.txt`
- `distance_3p1.txt`
- `heatmap_cp_3p1.txt`
- `heatmap_delta_pmumu_3p2.txt`
- `inverse_seesaw_3p1.txt`

## Presets principaux

- `config/presets/energy_3p1.txt`
- `config/presets/energy_3p2.txt`
- `config/presets/distance_3p1.txt`
- `config/presets/heatmap_cp_3p1.txt`
- `config/presets/heatmap_delta_pmumu_3p2.txt`
- `config/presets/inverse_seesaw_3p1.txt`

Exécution :

`bin/app.exe config/presets/energy_3p1.txt`