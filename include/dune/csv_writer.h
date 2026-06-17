#ifndef DUNE_CSV_WRITER_H
#define DUNE_CSV_WRITER_H

#include "dune/dune.h"

DuneStatus dune_csv_writer_write_prediction(const char *path, const DuneSamplePrediction *prediction);
DuneStatus dune_csv_writer_write_null(const char *path, const DuneSamplePrediction *prediction);
DuneStatus dune_csv_writer_write_residuals(const char *path, const DuneSamplePrediction *prediction);
DuneStatus dune_csv_writer_write_point_observables(const char *path, const DuneTheoryPoint *point);

#endif
