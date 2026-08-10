CC 			=	ccache clang++
APP 		= 	cough-e
BUILD_DIR 	= 	build
OBJ_DIR		=	$(BUILD_DIR)/obj
.DEFAULT_GOAL = all

SRC_DIRS 	= 	Src kiss_fftr FxP/imu FxP/audio
SRC 		= 	$(foreach c, $(SRC_DIRS), $(wildcard $c/*.c*))
OBJ			=	$(SRC:%=$(OBJ_DIR)/%.o)
MAIN_OBJ	=	$(OBJ_DIR)/main.cpp.o
DEPS		=	$(OBJ:.o=.d) $(MAIN_OBJ:.o=.d)

INC_DIRS 	= 	Inc FxP universal/include/sw libposit/include ./ kiss_fftr precision_analysis range_analysis
INC 		= 	$(foreach d, $(INC_DIRS), -I$d)

TWIDDLE_GENERATOR = kiss_fftr/gen_twiddles_fixed.py
TWIDDLE_Q15 = kiss_fftr/twiddles_win08_fs8000_q15.cpp
TWIDDLE_Q31 = kiss_fftr/twiddles_win08_fs8000_q31.cpp

MATH        =    -lm
# MATH        =    -Wl,--whole-archive /opt/newlib/x86_64-elf/lib/libm.a -Wl,--no-whole-archive \
# 				-ffp-contract=off -fno-builtin

CXXFLAGS	=	-w -O3 -MMD -MP -g -std=c++20 -stdlib=libc++ $(INC) $(MATH) -lc $(CFLAGS)

COMP_STRING = 	" |++++++++++++++++++++++++++++++++|\n"
COMP_STRING +=  "|            Cough-E             |\n"
COMP_STRING += 	"|      Compilation successful!   |\n"
COMP_STRING += 	"|++++++++++++++++++++++++++++++++|\n"
COMP_STRING += 	"|              ( (               |\n"
COMP_STRING += 	"|              ) )               |\n"
COMP_STRING += 	"|           ........             |\n"
COMP_STRING += 	"|           |      |]            |\n"
COMP_STRING += 	"|           \      /             |\n"
COMP_STRING += 	"|            '----'              |\n"
COMP_STRING += 	"|++++++++++++++++++++++++++++++++|\n"

EXEC_STRING =	" |++++++++++++++++++++++++++++++++|\n"
EXEC_STRING +=	"|     Start of execution...      |\n"
EXEC_STRING +=	"|++++++++++++++++++++++++++++++++|\n"


all: $(TWIDDLE_Q15) $(TWIDDLE_Q31) $(BUILD_DIR)/$(APP)
	@echo -e $(COMP_STRING)

$(BUILD_DIR)/$(APP): $(MAIN_OBJ) $(OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CC) -o $@ $^ $(CXXFLAGS)

$(MAIN_OBJ): main.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: % $(OBJ_DIR)/.cxxflags
	@mkdir -p $(dir $@)
	$(CC) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/.cxxflags: FORCE
	@mkdir -p $(dir $@)
	@echo '$(CXXFLAGS)' | cmp -s - $@ || echo '$(CXXFLAGS)' > $@

FORCE:

-include $(DEPS)

run: all
	@echo -e $(EXEC_STRING)
	@./$(BUILD_DIR)/$(APP)

$(TWIDDLE_Q15) $(TWIDDLE_Q31): $(TWIDDLE_GENERATOR)
	python3 $(TWIDDLE_GENERATOR) --formats q15 q31 --out-dir kiss_fftr

submodule:
	git submodule update --init
	git submodule add -f https://github.com/stillwater-sc/universal universal
	git submodule add -f https://github.com/CharlesWithC/libposit libposit
	cd universal && git sparse-checkout init --no-cone && git sparse-checkout set sparse-checkout set /include/sw
	cd libposit && git sparse-checkout init --no-cone && git sparse-checkout set sparse-checkout set /include

clean:
	rm -rf $(BUILD_DIR)
