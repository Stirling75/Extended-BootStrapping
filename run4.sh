g++ FD_update.cpp -o fdup1 -ltfhe-spqlios-fma -lquadmath -fopenmp
g++ FD_update2.cpp -o fdup2 -ltfhe-spqlios-fma -lquadmath


chmod 755 fdup1
chmod 755 fdup2

./fdup1 1 4 500
./fdup1 2 4 500
./fdup1 3 4 500
./fdup1 4 4 500
./fdup1 5 4 500
./fdup1 6 4 500
./fdup1 7 4 500
./fdup1 8 4 500

./fdup2 1 4 100
./fdup2 2 4 100
./fdup2 3 4 100
./fdup2 4 4 100
./fdup2 5 4 100
./fdup2 6 4 100
./fdup2 7 4 100
./fdup2 8 4 100
