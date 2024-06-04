export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

cd build
# cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --fresh ..
time make -j4




# cmake --fresh -DCREATE_DOCS=ON ..