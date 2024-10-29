
CC 			=	gcc
APP 		= 	cough-e

SRC_DIRS 	= 	Src kiss_fftr
SRC 		= 	$(foreach c, $(SRC_DIRS), $c/*)

INC_DIRS 	= 	Inc ./ kiss_fftr
INC 		= 	$(foreach d, $(INC_DIRS), -I$d)

MATH 		= 	-lm


all:
	$(CC) -Wall -o $(APP) main.c $(SRC) $(INC) $(MATH) -O3 -g
	@echo "Compilation terminated successfully! ☕\n"

run: all
	@echo "Running with command:"
	./$(APP)


profile:
	rm profile gmon.out
	$(CC) -Wall -pg -o profile main.c $(SRC) $(INC) $(MATH) -O3
	./profile
	gprof profile | gprof2dot | dot -Tpng -o prof_graph.png
