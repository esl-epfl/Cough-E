
CC 			=	gcc
APP 		= 	cough-e

SRC_DIRS 	= 	Src kiss_fftr
SRC 		= 	$(foreach c, $(SRC_DIRS), $c/*)

INC_DIRS 	= 	Inc ./ kiss_fftr
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
	$(CC) -Wall -o $(APP) main.c $(SRC) $(INC) $(MATH) -O3 -g
	@echo $(COMP_STRING)


run: all
	@echo $(EXEC_STRING)
	@./$(APP)
