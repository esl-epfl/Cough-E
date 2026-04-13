
CC 			=	gcc
APP 		= 	cough-e
BUILD_DIR 	= 	build

SRC_DIRS 	= 	Src kiss_fftr
SRC 		= 	$(foreach c, $(SRC_DIRS), $(wildcard $c/*.c))

INC_DIRS 	= 	Inc FxP ./ kiss_fftr
INC 		= 	$(foreach d, $(INC_DIRS), -I$d)

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


all:
	@mkdir -p $(BUILD_DIR)
	$(CC) -Wall -o $(BUILD_DIR)/$(APP) main.c $(SRC) $(INC) $(MATH) -O3 -g $(CFLAGS)
	@echo $(COMP_STRING)


run: all
	@echo $(EXEC_STRING)
	@./$(BUILD_DIR)/$(APP)

clean:
	rm -rf $(BUILD_DIR)
