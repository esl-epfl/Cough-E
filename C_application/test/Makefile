# Makefile for FxP IMU test harness
# Run from C_application/test/  OR  from C_application/ with:
#   make -C test

CC      = gcc
ROOT    = ..
SRCS    = $(wildcard $(ROOT)/Src/*.c) $(wildcard $(ROOT)/kiss_fftr/*.c) $(wildcard $(ROOT)/FxP/imu/*.c)
INC     = -I$(ROOT)/Inc -I$(ROOT)/FxP -I$(ROOT) -I$(ROOT)/kiss_fftr
CFLAGS  = -Wall -Wextra -Wno-unused-function -std=c11 -g -DFXP_MODE
LIBS    = -lm
PYTHON  = python3
.DEFAULT_GOAL = all

QFMT_MANIFEST = $(ROOT)/FxP/core/qformats.yaml
QFMT_GENERATOR = $(ROOT)/FxP/tools/generate_qformats.py
QFMT_HEADER = $(ROOT)/FxP/core/fxp_qformats.h

TARGET_IMU      = fxp_model_regression

$(QFMT_HEADER): $(QFMT_MANIFEST) $(QFMT_GENERATOR)
	$(PYTHON) $(QFMT_GENERATOR) $(QFMT_MANIFEST) $(QFMT_HEADER)

all: $(QFMT_HEADER) $(TARGET_IMU)

$(TARGET_IMU): fxp_model_regression.c $(SRCS) $(QFMT_HEADER)
	$(CC) $(CFLAGS) -o $@ fxp_model_regression.c $(SRCS) $(INC) $(LIBS)

run: all
	./$(TARGET_IMU)

run_regression: $(TARGET_IMU)
	./$(TARGET_IMU)

clean:
	rm -f $(TARGET_IMU)
