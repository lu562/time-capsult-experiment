all:
	g++ -g -O2 -std=c++17 -pthread -march=native matrix_mul.cpp -o matrix_mul -lntl -lgmp -lm
	cp matrix_mul /usr/local/bin/matrix_mul
clean:
	rm -f matrix_mul
