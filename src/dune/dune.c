#include "dune/dune.h"

#include <stdio.h>
#include <string.h>

const char *dune_status_message(DuneStatus status) {
    switch (status) {
        case DUNE_STATUS_OK:
            return "ok";
        case DUNE_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case DUNE_STATUS_MISSING_INPUT:
            return "missing physical input";
        default:
            return "unknown DUNE status";
    }
}

DuneStatus dune_run_context_init(DuneRunContext *ctx, const SimulationConfig *cfg) {
    if (!ctx || !cfg) {
        return DUNE_STATUS_INVALID_ARGUMENT;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->cfg = cfg;
    ctx->kernel.regime = DUNE_REGIME_UNKNOWN;
    return DUNE_STATUS_OK;
}
