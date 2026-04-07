# Makefile for FxP IMU test harness
# Run from C_application/test/  OR  from C_application/ with:
#   make -C test

CC      = gcc
ROOT    = ..
SRCS    = $(wildcard $(ROOT)/Src/*.c) $(wildcard $(ROOT)/kiss_fftr/*.c)
INC     = -I$(ROOT)/Inc -I$(ROOT) -I$(ROOT)/kiss_fftr
CFLAGS  = -Wall -Wextra -Wno-unused-function -std=c11 -g -DFXP_MODE
LIBS    = -lm

TARGET  = fxp_test_imu

all: $(TARGET)

$(TARGET): fxp_test_imu.c $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(INC) $(LIBS)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)
