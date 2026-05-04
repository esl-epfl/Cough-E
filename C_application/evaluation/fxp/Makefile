# Makefile for FxP validation/evaluation harnesses.
# Run from C_application/evaluation/fxp/ OR from the repository with:
#   make -C C_application/evaluation/fxp

CC      = gcc
ROOT    = ../..
SRCS    = $(wildcard $(ROOT)/Src/*.c) $(wildcard $(ROOT)/kiss_fftr/*.c) $(wildcard $(ROOT)/FxP/imu/*.c) $(wildcard $(ROOT)/FxP/audio/*.c)
FLOAT_SRCS = $(filter-out $(ROOT)/Src/fsm_control.c,$(wildcard $(ROOT)/Src/*.c)) $(wildcard $(ROOT)/kiss_fftr/*.c)
STAGE_FLOAT_SRCS = $(filter-out $(ROOT)/Src/feature_extraction.c,$(FLOAT_SRCS))
INC     = -I$(ROOT)/Inc -I$(ROOT)/FxP -I$(ROOT) -I$(ROOT)/kiss_fftr -I$(ROOT)/range_analysis
CFLAGS  = -Wall -Wextra -Wno-unused-function -std=c11 -g
LIBS    = -lm
.DEFAULT_GOAL = all

FXP_CORE_HEADER = $(ROOT)/FxP/core/fxp_core.h

TARGET_STAGE = fxp_stage_harness
TARGET_PROGRESSIVE = fxp_progressive_harness
FFT_MODE     ?= -DFIXED_POINT=32
HEADER_FLAGS ?=
BUILD_TAG    ?= default
OBJDIR       = .build/$(BUILD_TAG)
STAGE_PROBE_FLAGS = -DFXP_STAGE_PROBES
STAGE_BIN    = $(OBJDIR)/$(TARGET_STAGE)
PROGRESSIVE_BIN = $(OBJDIR)/$(TARGET_PROGRESSIVE)

STAGE_OBJ = $(OBJDIR)/fxp_stage_harness.o
STAGE_FXP_OBJS = \
	$(OBJDIR)/fxp_feature_extraction_stage.o \
	$(OBJDIR)/fxp_audio_pipeline_stage.o \
	$(OBJDIR)/fxp_imu_pipeline_stage.o \
	$(OBJDIR)/fxp_kiss_fft_stage.o \
	$(OBJDIR)/fxp_kiss_fftr_stage.o
PROGRESSIVE_OBJS = \
	$(OBJDIR)/fxp_progressive_harness.o \
	$(OBJDIR)/fxp_audio_pipeline_prog.o \
	$(OBJDIR)/fxp_imu_pipeline_prog.o \
	$(OBJDIR)/fxp_kiss_fft_prog.o \
	$(OBJDIR)/fxp_kiss_fftr_prog.o

.PHONY: all run clean $(TARGET_STAGE) $(TARGET_PROGRESSIVE)

all: $(TARGET_STAGE)

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

FXP_FEATURE_RENAME = \
	-Daudio_features=fxp_audio_features \
	-Dimu_features=fxp_imu_features

$(TARGET_STAGE): $(STAGE_BIN)

$(STAGE_BIN): fxp_stage_harness.c fxp_metrics.h $(STAGE_FLOAT_SRCS) $(ROOT)/Src/feature_extraction.c $(ROOT)/FxP/audio/audio_pipeline_fxp.c $(ROOT)/FxP/imu/imu_pipeline.c $(ROOT)/kiss_fftr/kiss_fft.c $(ROOT)/kiss_fftr/kiss_fftr.c $(FXP_CORE_HEADER)
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -DFXP_MODE $(STAGE_PROBE_FLAGS) $(FFT_MODE) $(HEADER_FLAGS) -c -o $(STAGE_OBJ) fxp_stage_harness.c $(INC)
	$(CC) $(CFLAGS) -DFXP_MODE $(FFT_MODE) $(FXP_FEATURE_RENAME) $(KISS_RENAME) -c -o $(OBJDIR)/fxp_feature_extraction_stage.o $(ROOT)/Src/feature_extraction.c $(INC)
	$(CC) $(CFLAGS) -DFXP_MODE $(STAGE_PROBE_FLAGS) $(FFT_MODE) $(KISS_RENAME) -c -o $(OBJDIR)/fxp_audio_pipeline_stage.o $(ROOT)/FxP/audio/audio_pipeline_fxp.c $(INC)
	$(CC) $(CFLAGS) -DFXP_MODE $(FFT_MODE) -c -o $(OBJDIR)/fxp_imu_pipeline_stage.o $(ROOT)/FxP/imu/imu_pipeline.c $(INC)
	$(CC) $(CFLAGS) $(FFT_MODE) $(KISS_RENAME) -c -o $(OBJDIR)/fxp_kiss_fft_stage.o $(ROOT)/kiss_fftr/kiss_fft.c $(INC)
	$(CC) $(CFLAGS) $(FFT_MODE) $(KISS_RENAME) -c -o $(OBJDIR)/fxp_kiss_fftr_stage.o $(ROOT)/kiss_fftr/kiss_fftr.c $(INC)
	$(CC) $(CFLAGS) -o $@ $(STAGE_OBJ) $(STAGE_FXP_OBJS) $(STAGE_FLOAT_SRCS) $(INC) $(LIBS)

$(TARGET_PROGRESSIVE): $(PROGRESSIVE_BIN)

$(PROGRESSIVE_BIN): fxp_progressive_harness.c $(FLOAT_SRCS) $(ROOT)/FxP/audio/audio_pipeline_fxp.c $(ROOT)/FxP/imu/imu_pipeline.c $(ROOT)/kiss_fftr/kiss_fft.c $(ROOT)/kiss_fftr/kiss_fftr.c $(FXP_CORE_HEADER)
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(HEADER_FLAGS) -c -o $(OBJDIR)/fxp_progressive_harness.o fxp_progressive_harness.c $(INC)
	$(CC) $(CFLAGS) -DFXP_MODE $(FFT_MODE) $(KISS_RENAME) -c -o $(OBJDIR)/fxp_audio_pipeline_prog.o $(ROOT)/FxP/audio/audio_pipeline_fxp.c $(INC)
	$(CC) $(CFLAGS) -DFXP_MODE $(FFT_MODE) -c -o $(OBJDIR)/fxp_imu_pipeline_prog.o $(ROOT)/FxP/imu/imu_pipeline.c $(INC)
	$(CC) $(CFLAGS) $(FFT_MODE) $(KISS_RENAME) -c -o $(OBJDIR)/fxp_kiss_fft_prog.o $(ROOT)/kiss_fftr/kiss_fft.c $(INC)
	$(CC) $(CFLAGS) $(FFT_MODE) $(KISS_RENAME) -c -o $(OBJDIR)/fxp_kiss_fftr_prog.o $(ROOT)/kiss_fftr/kiss_fftr.c $(INC)
	$(CC) $(CFLAGS) -o $@ $(PROGRESSIVE_OBJS) $(FLOAT_SRCS) $(INC) $(LIBS)

run: all
	./$(STAGE_BIN)

clean:
	rm -rf fxp_stage_harness fxp_stage_harness_* fxp_progressive_harness fxp_progressive_harness_* *.dSYM .build
