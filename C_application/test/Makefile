# Makefile for FxP regression harnesses (IMU + audio FFT + audio periodogram)
# Run from C_application/test/  OR  from C_application/ with:
#   make -C test

CC      = gcc
ROOT    = ..
SRCS    = $(wildcard $(ROOT)/Src/*.c) $(wildcard $(ROOT)/kiss_fftr/*.c) $(wildcard $(ROOT)/FxP/imu/*.c) $(wildcard $(ROOT)/FxP/audio/*.c)
INC     = -I$(ROOT)/Inc -I$(ROOT)/FxP -I$(ROOT) -I$(ROOT)/kiss_fftr -I$(ROOT)/range_analysis
CFLAGS  = -Wall -Wextra -Wno-unused-function -std=c11 -g -DFXP_MODE
LIBS    = -lm
PYTHON  = python3
.DEFAULT_GOAL = all

QFMT_MANIFEST = $(ROOT)/FxP/core/qformats.yaml
QFMT_GENERATOR = $(ROOT)/FxP/tools/generate_qformats.py
QFMT_HEADER = $(ROOT)/FxP/core/fxp_qformats.h

TARGET_IMU      = fxp_model_regression
TARGET_AUDIO_FFT = fxp_audio_fft_regression
TARGET_AUDIO_PSD = fxp_audio_periodogram_regression
TARGET_AUDIO_MEL = fxp_audio_mel_regression
FFT_MODE ?= -DFIXED_POINT=16

$(QFMT_HEADER): $(QFMT_MANIFEST) $(QFMT_GENERATOR)
	$(PYTHON) $(QFMT_GENERATOR) $(QFMT_MANIFEST) $(QFMT_HEADER)

all: $(QFMT_HEADER) $(TARGET_IMU) $(TARGET_AUDIO_FFT) $(TARGET_AUDIO_PSD) $(TARGET_AUDIO_MEL)

$(TARGET_IMU): fxp_model_regression.c $(SRCS) $(QFMT_HEADER)
	$(CC) $(CFLAGS) -o $@ fxp_model_regression.c $(SRCS) $(INC) $(LIBS)

$(TARGET_AUDIO_FFT): fxp_audio_fft_regression.c $(SRCS) $(QFMT_HEADER)
	$(CC) $(CFLAGS) $(FFT_MODE) -o $@ fxp_audio_fft_regression.c $(SRCS) $(INC) $(LIBS)

$(TARGET_AUDIO_PSD): fxp_audio_periodogram_regression.c $(SRCS) $(QFMT_HEADER)
	$(CC) $(CFLAGS) $(FFT_MODE) -o $@ fxp_audio_periodogram_regression.c $(SRCS) $(INC) $(LIBS)

$(TARGET_AUDIO_MEL): fxp_audio_mel_regression.c $(SRCS) $(QFMT_HEADER)
	$(CC) $(CFLAGS) $(FFT_MODE) -o $@ fxp_audio_mel_regression.c $(SRCS) $(INC) $(LIBS)

run: all
	./$(TARGET_IMU)

run_regression: $(TARGET_IMU)
	./$(TARGET_IMU)

run_audio_fft16: FFT_MODE=-DFIXED_POINT=16
run_audio_fft16: $(TARGET_AUDIO_FFT)
	./$(TARGET_AUDIO_FFT)

run_audio_fft32: FFT_MODE=-DFIXED_POINT=32
run_audio_fft32: $(TARGET_AUDIO_FFT)
	./$(TARGET_AUDIO_FFT)

run_audio_psd16: FFT_MODE=-DFIXED_POINT=16
run_audio_psd16: $(TARGET_AUDIO_PSD)
	./$(TARGET_AUDIO_PSD)

run_audio_psd32: FFT_MODE=-DFIXED_POINT=32
run_audio_psd32: $(TARGET_AUDIO_PSD)
	./$(TARGET_AUDIO_PSD)

run_audio_mel16: FFT_MODE=-DFIXED_POINT=16
run_audio_mel16: $(TARGET_AUDIO_MEL)
	./$(TARGET_AUDIO_MEL)

run_audio_mel32: FFT_MODE=-DFIXED_POINT=32
run_audio_mel32: $(TARGET_AUDIO_MEL)
	./$(TARGET_AUDIO_MEL)

clean:
	rm -f $(TARGET_IMU) $(TARGET_AUDIO_FFT) $(TARGET_AUDIO_PSD) $(TARGET_AUDIO_MEL)
