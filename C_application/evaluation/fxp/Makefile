# Makefile for FxP validation/evaluation harnesses.
# Run from C_application/evaluation/fxp/ OR from the repository with:
#   make -C C_application/evaluation/fxp

CC      = gcc
ROOT    = ../..
SRCS    = $(wildcard $(ROOT)/Src/*.c) $(wildcard $(ROOT)/kiss_fftr/*.c) $(wildcard $(ROOT)/FxP/imu/*.c) $(wildcard $(ROOT)/FxP/audio/*.c)
FLOAT_SRCS = $(filter-out $(ROOT)/Src/fsm_control.c,$(wildcard $(ROOT)/Src/*.c)) $(wildcard $(ROOT)/kiss_fftr/*.c)
INC     = -I$(ROOT)/Inc -I$(ROOT)/FxP -I$(ROOT) -I$(ROOT)/kiss_fftr -I$(ROOT)/range_analysis
CFLAGS  = -Wall -Wextra -Wno-unused-function -std=c11 -g
LIBS    = -lm
.DEFAULT_GOAL = all

QFMT_HEADER = $(ROOT)/FxP/core/fxp_qformats.h

TARGET_STAGE = fxp_stage_harness
TARGET_PROGRESSIVE = fxp_progressive_harness
FFT_MODE     ?= -DFIXED_POINT=16
HEADER_FLAGS ?=

all: $(TARGET_STAGE)

$(TARGET_STAGE): fxp_stage_harness.c fxp_metrics.h $(SRCS) $(QFMT_HEADER)
	$(CC) $(CFLAGS) -DFXP_MODE $(FFT_MODE) $(HEADER_FLAGS) -c -o fxp_stage_harness.o fxp_stage_harness.c $(INC)
	$(CC) $(CFLAGS) -DFXP_MODE $(FFT_MODE) -o $@ fxp_stage_harness.o $(SRCS) $(INC) $(LIBS)
	rm -f fxp_stage_harness.o

KISS_RENAME = \
	-Dkiss_fft_alloc=fxp_kiss_fft_alloc \
	-Dkiss_fft=fxp_kiss_fft \
	-Dkiss_fft_stride=fxp_kiss_fft_stride \
	-Dkiss_fft_cleanup=fxp_kiss_fft_cleanup \
	-Dkiss_fft_next_fast_size=fxp_kiss_fft_next_fast_size \
	-Dinit_twiddles=fxp_init_twiddles \
	-Dkiss_fftr_alloc=fxp_kiss_fftr_alloc \
	-Dkiss_fftr=fxp_kiss_fftr \
	-Dkiss_fftri=fxp_kiss_fftri \
	-Dinit_super_twiddles=fxp_init_super_twiddles \
	-Dtostring=fxp_tostring

$(TARGET_PROGRESSIVE): fxp_progressive_harness.c $(FLOAT_SRCS) $(ROOT)/FxP/audio/audio_pipeline_fxp.c $(ROOT)/FxP/imu/imu_pipeline.c $(ROOT)/kiss_fftr/kiss_fft.c $(ROOT)/kiss_fftr/kiss_fftr.c $(QFMT_HEADER)
	$(CC) $(CFLAGS) $(HEADER_FLAGS) -c -o fxp_progressive_harness.o fxp_progressive_harness.c $(INC)
	$(CC) $(CFLAGS) -DFXP_MODE $(FFT_MODE) $(KISS_RENAME) -c -o fxp_audio_pipeline_prog.o $(ROOT)/FxP/audio/audio_pipeline_fxp.c $(INC)
	$(CC) $(CFLAGS) -DFXP_MODE $(FFT_MODE) -c -o fxp_imu_pipeline_prog.o $(ROOT)/FxP/imu/imu_pipeline.c $(INC)
	$(CC) $(CFLAGS) $(FFT_MODE) $(KISS_RENAME) -c -o fxp_kiss_fft_prog.o $(ROOT)/kiss_fftr/kiss_fft.c $(INC)
	$(CC) $(CFLAGS) $(FFT_MODE) $(KISS_RENAME) -c -o fxp_kiss_fftr_prog.o $(ROOT)/kiss_fftr/kiss_fftr.c $(INC)
	$(CC) $(CFLAGS) -o $@ fxp_progressive_harness.o fxp_audio_pipeline_prog.o fxp_imu_pipeline_prog.o fxp_kiss_fft_prog.o fxp_kiss_fftr_prog.o $(FLOAT_SRCS) $(INC) $(LIBS)
	rm -f fxp_progressive_harness.o fxp_audio_pipeline_prog.o fxp_imu_pipeline_prog.o fxp_kiss_fft_prog.o fxp_kiss_fftr_prog.o

run: all
	./$(TARGET_STAGE)

clean:
	rm -rf $(TARGET_STAGE) $(TARGET_STAGE).dSYM $(TARGET_PROGRESSIVE) $(TARGET_PROGRESSIVE).dSYM
