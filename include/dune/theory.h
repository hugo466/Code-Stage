#ifndef DUNE_THEORY_H
#define DUNE_THEORY_H

#include "dune/dune.h"

typedef struct {
    int point_id;
    char model[32];
    char point_file[512];
    int n_light;
    int n_heavy;
    char mode_hint[32];
    double max_abs_eta;
    double dm41_eV2;
    double dm51_eV2;
} DuneTheoryIndexEntry;

DuneStatus dune_theory_index_find_entry(const char *index_csv, int point_id, DuneTheoryIndexEntry *entry);
DuneStatus dune_theory_index_find_point(const char *index_csv, int point_id, char *path, int path_size);
DuneStatus dune_iss23_read_point(const char *path, DuneTheoryPoint *point);
DuneStatus dune_theory_point_load(DuneRunContext *ctx);

#endif
