#ifndef DUNE_FLUX_H
#define DUNE_FLUX_H

#include "dune/dune.h"

#define DUNE_FLUX_TABLE_MAX_POINTS 16384

typedef enum {
    DUNE_BEAM_FHC = 0,
    DUNE_BEAM_RHC
} DuneBeamMode;

typedef enum {
    DUNE_FLUX_NUMU = 0,
    DUNE_FLUX_NUMUBAR,
    DUNE_FLUX_NUE,
    DUNE_FLUX_NUEBAR,
    DUNE_FLUX_NUTAU,
    DUNE_FLUX_NUTAUBAR
} DuneFluxFlavor;

typedef enum {
    DUNE_SOURCE_GEOMETRY_FIXED = 0,
    DUNE_SOURCE_GEOMETRY_SOURCE_LINE
} DuneSourceGeometryModel;

typedef struct {
    int n_points;
    double energy_GeV[DUNE_FLUX_TABLE_MAX_POINTS];
    double flux[6][DUNE_FLUX_TABLE_MAX_POINTS];
} DuneFluxTable;

typedef struct {
    DuneSourceGeometryModel model;
    double baseline_km;
    double source_z_min_m;
    double source_z_max_m;
    int n_source_steps;
} DuneSourceGeometry;

double dune_flux_analytic_test(double energy_GeV);
DuneStatus dune_flux_table_load_globes(const char *path, DuneFluxTable *table);
DuneStatus dune_flux_table_eval(
    const DuneFluxTable *table,
    DuneFluxFlavor flavor,
    double energy_GeV,
    double *flux);
DuneStatus dune_flux_provider_load(DuneRunContext *ctx);
DuneStatus dune_flux_provider_eval(
    const DuneRunContext *ctx,
    DuneFluxFlavor flavor,
    double energy_GeV,
    double source_z_m,
    double *flux);
DuneStatus dune_source_geometry_baseline(
    const DuneSourceGeometry *geometry,
    double source_z_m,
    double *baseline_km);

#endif
