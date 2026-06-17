#ifndef BEAM_DK2NU_FLUX_Z_H
#define BEAM_DK2NU_FLUX_Z_H

#include "dune/dune.h"
#include "dune/flux.h"

typedef struct {
    DuneFluxFlavor flavor;
    double e_low_GeV;
    double e_high_GeV;
    double z_low_m;
    double z_high_m;
    double weight;
} DuneDk2nuFluxZRow;

typedef struct {
    DuneDk2nuFluxZRow *rows;
    int n_rows;
    int capacity;
    double e_min_GeV;
    double e_max_GeV;
    double z_min_m;
    double z_max_m;
} DuneDk2nuFluxZTable;

DuneStatus dune_dk2nu_flux_z_load_csv(const char *path, DuneDk2nuFluxZTable *table);
void dune_dk2nu_flux_z_free(DuneDk2nuFluxZTable *table);
double dune_dk2nu_flux_z_weight_sum(
    const DuneDk2nuFluxZTable *table,
    DuneFluxFlavor flavor,
    double energy_GeV);

#endif
