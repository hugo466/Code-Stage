# Organisation des figures DUNE ND

- `reference/` : spectres de reference ND, sans point ISS specifique.
- `flux/` : flux ND, profils de source dk2nu et validations associees.
- `detector_response/` : matrices de smearing et reponses detecteur ND.
- `iss23/` : figures ISS(2,3), donc cadre 3+1. Les points sont ranges comme `construct23_pointXX/`.
- `iss24/` : figures ISS(2,4), donc cadre 3+2. Les points sont ranges comme `construct24_pointXX/`.

Convention pour les points :

```text
iss23/construct23_point70/fig4/
iss23/construct23_point70/probabilities/
iss23/construct23_point70/source_models/
iss24/construct24_pointXX/fig4/
```

Les cartes globales et exclusions ISS(2,3) sont rangees dans :

```text
iss23/scan_maps/
iss23/exclusion/
```
