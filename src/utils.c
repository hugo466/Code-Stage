#include "utils.h"

#include <direct.h>

double clamp_probability(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

int ensure_directory_exists(const char *path) {
    return _mkdir(path);
}