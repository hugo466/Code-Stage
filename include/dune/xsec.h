#ifndef DUNE_XSEC_H
#define DUNE_XSEC_H

#include "dune/dune.h"
#include "dune/flux.h"

#define DUNE_XSEC_TABLE_MAX_POINTS 16384

typedef struct {
    int n_points;
    double energy_GeV[DUNE_XSEC_TABLE_MAX_POINTS];
    double xsec[6][DUNE_XSEC_TABLE_MAX_POINTS];
} DuneXsecTable;

DuneStatus dune_xsec_table_load_globes(const char *path, DuneXsecTable *table);
DuneStatus dune_xsec_table_eval(
    const DuneXsecTable *table,
    DuneFluxFlavor flavor,
    double energy_GeV,
    double *xsec);
DuneStatus dune_xsec_model_load_from_config(const SimulationConfig *cfg);
DuneStatus dune_xsec_model_eval(
    DuneFluxFlavor flavor,
    int use_cc,
    double energy_GeV,
    double *xsec);

#endif
