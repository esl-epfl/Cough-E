# Makefile for the unified FxP validation harness.
# Run from C_application/evaluation/fxp/ OR from the repository with:
#   make -C C_application/evaluation/fxp

CC      = gcc
ROOT    = ../..
SRCS    = $(wildcard $(ROOT)/Src/*.c) $(wildcard $(ROOT)/kiss_fftr/*.c) $(wildcard $(ROOT)/FxP/imu/*.c) $(wildcard $(ROOT)/FxP/audio/*.c)
INC     = -I$(ROOT)/Inc -I$(ROOT)/FxP -I$(ROOT) -I$(ROOT)/kiss_fftr -I$(ROOT)/range_analysis
CFLAGS  = -Wall -Wextra -Wno-unused-function -std=c11 -g -DFXP_MODE
LIBS    = -lm
PYTHON  = python3
.DEFAULT_GOAL = all

QFMT_MANIFEST = $(ROOT)/FxP/core/qformats.yaml
QFMT_GENERATOR = $(ROOT)/FxP/tools/generate_qformats.py
QFMT_HEADER = $(ROOT)/FxP/core/fxp_qformats.h

TARGET_STAGE     = fxp_stage_harness
FFT_MODE ?= -DFIXED_POINT=16

$(QFMT_HEADER): $(QFMT_MANIFEST) $(QFMT_GENERATOR)
	$(PYTHON) $(QFMT_GENERATOR) $(QFMT_MANIFEST) $(QFMT_HEADER)

all: $(QFMT_HEADER) $(TARGET_STAGE)

$(TARGET_STAGE): fxp_stage_harness.c fxp_metrics.h $(SRCS) $(QFMT_HEADER)
	$(CC) $(CFLAGS) $(FFT_MODE) -o $@ fxp_stage_harness.c $(SRCS) $(INC) $(LIBS)

run: all
	./$(TARGET_STAGE)

clean:
	rm -f $(TARGET_STAGE)
