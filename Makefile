CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -Iinclude
LDFLAGS = -lm
PYTHON = python
CONFIG ?= config.txt
PLOT ?= plots/plot_probabilities.py

TARGET = bin/app.exe
TARGET_CMD = bin\app.exe
SOURCES = $(wildcard src/*.c)

.PHONY: all build rebuild run plot clean

all: build

build:
	@if not exist bin mkdir bin
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)

rebuild: clean build

run: build
	$(TARGET_CMD) $(CONFIG)

plot: run
	$(PYTHON) $(PLOT)

clean:
	@if exist $(TARGET_CMD) del /Q $(TARGET_CMD)
