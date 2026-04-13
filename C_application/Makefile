
CC 			=	gcc
APP 		= 	cough-e
BUILD_DIR 	= 	build
PYTHON      =   python3
.DEFAULT_GOAL = all

SRC_DIRS 	= 	Src kiss_fftr FxP/imu
SRC 		= 	$(foreach c, $(SRC_DIRS), $(wildcard $c/*.c))

INC_DIRS 	= 	Inc FxP ./ kiss_fftr
INC 		= 	$(foreach d, $(INC_DIRS), -I$d)

QFMT_MANIFEST = FxP/core/qformats.yaml
QFMT_GENERATOR = FxP/tools/generate_qformats.py
QFMT_HEADER = FxP/core/fxp_qformats.h

MATH 		= 	-lm

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

# @echo "Compilation terminated successfully! ☕\n"
# @echo "Running with command:"


EXEC_STRING =	" |++++++++++++++++++++++++++++++++|\n" 
EXEC_STRING +=	"|     Start of execution...      |\n" 
EXEC_STRING +=	"|++++++++++++++++++++++++++++++++|\n" 


$(QFMT_HEADER): $(QFMT_MANIFEST) $(QFMT_GENERATOR)
	$(PYTHON) $(QFMT_GENERATOR) $(QFMT_MANIFEST) $(QFMT_HEADER)

all: $(QFMT_HEADER)
	@mkdir -p $(BUILD_DIR)
	$(CC) -Wall -o $(BUILD_DIR)/$(APP) main.c $(SRC) $(INC) $(MATH) -O3 -g $(CFLAGS)
	@echo $(COMP_STRING)


run: all
	@echo $(EXEC_STRING)
	@./$(BUILD_DIR)/$(APP)

clean:
	rm -rf $(BUILD_DIR)
