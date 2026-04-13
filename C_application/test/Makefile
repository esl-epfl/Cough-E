# Makefile for FxP IMU test harness
# Run from C_application/test/  OR  from C_application/ with:
#   make -C test

CC      = gcc
ROOT    = ..
SRCS    = $(wildcard $(ROOT)/Src/*.c) $(wildcard $(ROOT)/kiss_fftr/*.c)
INC     = -I$(ROOT)/Inc -I$(ROOT)/FxP -I$(ROOT) -I$(ROOT)/kiss_fftr
CFLAGS  = -Wall -Wextra -Wno-unused-function -std=c11 -g -DFXP_MODE
LIBS    = -lm

TARGET          = fxp_test_imu
TARGET_KFFT     = kissfft_fxp_verify
TARGET_IMPULSE  = kissfft_fxp_impulse

all: $(TARGET)

$(TARGET): fxp_test_imu.c $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(INC) $(LIBS)

# KissFFT fixed-point readiness verification.
# Compiled without -DFXP_MODE: this is a pure float build that checks
# (a) kiss_fftr_alloc succeeds for each Cough-E FFT size,
# (b) the output dynamic range matches the Q-formats in OVERLEAF.md, and
# (c) Q2.14 input quantisation error propagates within the expected bound.
# Does NOT use -DFIXED_POINT (incompatible with this KissFFT fork — see harness).
$(TARGET_KFFT): kissfft_fxp_verify.c $(SRCS)
	$(CC) -Wall -Wextra -Wno-unused-function -std=c11 -g \
	    -o $@ $^ $(INC) $(LIBS)

run: all
	./$(TARGET)

run_kfft: $(TARGET_KFFT)
	./$(TARGET_KFFT)

# Impulse test: verifies FIXED_POINT=16 integer butterfly produces flat spectrum
# for a unit impulse, with magnitude error within Q2.14 quantisation tolerance.
$(TARGET_IMPULSE): kissfft_fxp_impulse.c $(ROOT)/kiss_fftr/kiss_fft.c $(ROOT)/kiss_fftr/kiss_fftr.c
	$(CC) -Wall -Wextra -Wno-unused-function -std=c11 -g -DFIXED_POINT=16 \
	    -o $@ $^ $(INC) $(LIBS)

run_impulse: $(TARGET_IMPULSE)
	./$(TARGET_IMPULSE)

clean:
	rm -f $(TARGET) $(TARGET_KFFT) $(TARGET_IMPULSE)
