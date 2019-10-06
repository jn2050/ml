LIBPATH=~/dev/lib
DEST=~/dev/lib/ml/lib

#nn
rm -rf $DEST/nn
mkdir -p $DEST/nn/nn
cp $LIBPATH/nn/setup.py $DEST/nn/setup.py
cp $LIBPATH/nn/src/export/*.py $DEST/nn/nn
