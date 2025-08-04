DIR=$(pwd)

which make
cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_MAKE_PROGRAM=$(which make) && make -j11
cd /kaolin && rm -rf build *egg* && pip install -e .
cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

cd ${DIR}
