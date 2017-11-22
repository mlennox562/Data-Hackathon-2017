#!/bin/sh

#chmod +x run.sh
#./run.sh


prepare() {
  echo "Preparing $1..."
  python prepare-$1.py
}

train() {
  echo "Training $1..."
  time python -u train.py --threads 4 $1 | tee logs/$1.log
}


# Prepare features

prepare basic
#prepare numeric-boxcox
#prepare numeric-scaled
#prepare numeric-rank-norm
#prepare categorical-encoded
#prepare categorical-counts

#prepare categorical-dummy #Leave this script out.
#prepare svd #Leave this script out.

## Basic models
#train lr-ce
#
#train et-ce
#train rf-ce
#train gb-ce
#
## LibFM
#train libfm-ce #Don't have it installed!
#
## LightGBM
#train lgb-ce
#
## XGB
#train xgb-ce
#train xgb-ce-2
#
#train xgbf-ce
#train xgbf-ce-2
