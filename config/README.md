# Configurations

- `config/base/` : blocs communs inclus par les presets.
- `config/presets/` : configurations prêtes à lancer, rangées par domaine.

## Arborescence

- `config/base/oscillations/` : paramètres communs 3+1 / 3+2.
- `config/base/dune/` : paramètres communs DUNE.
- `config/presets/oscillations/3p1/` : scans énergie, distance, CP.
- `config/presets/oscillations/3p2/` : scans 3+2.
- `config/presets/inverse_seesaw/3p1/` : filtres et construction ISS(2,3).
- `config/presets/inverse_seesaw/3p2/` : filtres 3+2.
- `config/presets/dune/nd/` : prédictions et validations ND.
- `config/presets/dune/fd/` : validations FD.

## Exemples

```bat
bin\app.exe config\presets\oscillations\3p1\energy.txt
bin\app.exe config\presets\inverse_seesaw\3p1\construct_23.txt
bin\app.exe config\presets\dune\nd\minimal_onaxis.ini
bin\app.exe config\presets\dune\fd\fig4_point70.ini
```
