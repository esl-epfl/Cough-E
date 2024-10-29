
CC 			=	gcc
APP 		= 	cough

SRC_DIRS 	= 	Src kiss_fftr
SRC 		= 	$(foreach c, $(SRC_DIRS), $c/*)

INC_DIRS 	= 	Inc ./ kiss_fftr
INC 		= 	$(foreach d, $(INC_DIRS), -I$d)

MATH 		= 	-lm

# To select which feature family to extract in `test-features`
FEAT 		:= 	IMU

all:
	$(CC) -Wall -o $(APP) main.c $(SRC) $(INC) $(MATH) -O3 -g
	@echo "Compilation terminated successfully!\n"

run: all
	@echo "Running with command:"
	./$(APP)


profile_memory:
	$(CC) -Wall -o $(APP) main.c $(SRC) $(INC) -I./profile_mem $(MATH) -O3
	@echo "Compilation terminated successfully!\n"
	@echo "Running with command:"
	./$(APP)

# test-feat:
# 	$(CC) -Wall -o $(APP) tester.c $(SRC) $(INC) $(MATH) -O3
# 	@echo "Compilation terminated successfully!\n"
# 	@echo "Running with command:"
# 	./$(APP)

profile:
	rm profile gmon.out
	$(CC) -Wall -pg -o profile main.c $(SRC) $(INC) $(MATH) -O3
	./profile
	gprof profile | gprof2dot | dot -Tpng -o prof_graph.png

prof:
	$(CC) -Wall -pg -o profile main.c $(SRC) $(INC) $(MATH) -O3
	./profile
	gprof profile | gprof2dot | dot -Tpng -o prof_graph.png

test-features:
	$(CC) -Wall -o $(APP) test_features.c $(SRC) $(INC) $(MATH) -O3 -D$(FEAT)
	./$(APP)

compare_features:
	$(CC) -Wall -o $(APP) test_features.c $(SRC) $(INC) $(MATH) -O3 -D$(FEAT)
	@echo "\nCompilation terminated successfully!\n"
	./$(APP) > ./feature_testing/out_feat.txt
	@echo "\nApp executed\n"
	python feature_testing/compare_features.py


dump:
	$(CC) -Wall -S build/ main.c $(SRC) $(INC) $(MATH) -O3