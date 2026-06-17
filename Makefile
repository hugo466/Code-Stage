SHELL := cmd
.SHELLFLAGS := /C

CC = gcc
PYTHON = python

CFLAGS ?= -Wall -Wextra -std=c11 -Iinclude
LDFLAGS ?= -lm

CONFIG ?= config.txt
ARGS ?=
PLOT ?= plots/plot_probabilities.py

TARGET_DIR := bin
TARGET := $(TARGET_DIR)\app.exe
SOURCES := $(filter-out \
	src/scan/scan_dune_fd_fig4.c \
	src/scan/scan_dune_nd.c \
	src/scan/scan_dune_nd_fig4.c, \
	$(wildcard src/*.c src/*/*.c))

.PHONY: all help build rebuild run plot compare-nd-sources plot-nd-source-comparison plot-optimized-sources clean distclean list-presets print-vars

all: build

help:
	@echo Commandes disponibles:
	@echo   mingw32-make build                         Compile $(TARGET)
	@echo   mingw32-make run CONFIG=chemin\preset.txt  Compile puis lance avec un preset
	@echo   mingw32-make plot PLOT=chemin\plot.py      Lance run puis le script Python choisi
	@echo   mingw32-make compare-nd-sources             Compare point/uniform/dk2nu pour le ND point 70
	@echo   mingw32-make plot-nd-source-comparison      Trace point/uniform/dk2nu pour le ND point 70
	@echo   mingw32-make plot-optimized-sources         Trace point opt / line opt / dk2nu
	@echo   mingw32-make rebuild                       Clean puis build
	@echo   mingw32-make clean                         Supprime l'executable compile
	@echo   mingw32-make distclean                     Supprime le dossier bin
	@echo   mingw32-make list-presets                  Liste les presets disponibles
	@echo   mingw32-make print-vars                    Affiche la config Makefile

build:
	@if not exist "$(TARGET_DIR)" mkdir "$(TARGET_DIR)"
	$(CC) $(CFLAGS) $(SOURCES) -o "$(TARGET)" $(LDFLAGS)

rebuild: clean build

run: build
	"$(TARGET)" "$(CONFIG)" $(ARGS)

plot: run
	$(PYTHON) "$(PLOT)"

compare-nd-sources:
	$(PYTHON) tools\compare_nd_source_models.py

plot-nd-source-comparison:
	$(PYTHON) plots\dune_nd\plot_nd_source_model_comparison.py

plot-optimized-sources:
	$(PYTHON) plots\dune_nd\plot_optimized_sources_vs_dk2nu.py

clean:
	@if exist "$(TARGET)" del /Q "$(TARGET)"

distclean:
	@if exist "$(TARGET_DIR)" rmdir /S /Q "$(TARGET_DIR)"

list-presets:
	@dir /S /B config\presets\*.txt config\presets\*.ini

print-vars:
	@echo CC=$(CC)
	@echo CFLAGS=$(CFLAGS)
	@echo LDFLAGS=$(LDFLAGS)
	@echo CONFIG=$(CONFIG)
	@echo ARGS=$(ARGS)
	@echo PLOT=$(PLOT)
	@echo TARGET=$(TARGET)
	@echo SOURCES=$(SOURCES)
