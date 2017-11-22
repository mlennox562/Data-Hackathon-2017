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
#train lr-cd
#train lr-cd-2
#train lr-cd-nr
#train ab-ce
#train et-ce
#train et-ce-2
#train et-ce-3
#train rf-ce
#train rf-ce-2
#train gb-ce
#
## LibFM
#train libfm-ce #Don't have it installed!
#
## LightGBM
#train lgb-ce
#train lgb-tst
#train lgb-cd-2
#train lgb-cd-tst
#
## XGB
#train xgb-tst
#train xgb2
#train xgb3
#train xgb4
#train xgb-ce
#train xgb-ce-2
#
#train xgbf-ce
#train xgbf-ce-2
#
## NNET
#train nn-tst
#train nn-1
#train nn-cd
#train nn-cd-2
#train nn-cd-3
#train nn-cd-clrbf
#train nn-svd
#train nn-svd-2

#train knn1

#train l2-nn
#train l2-nn-2
#train l2-nn-3
#train l2-nn-4
#train l2-xgbf
#train l2-xgbf-2
#train l2-xgbf-3
#train l2-xgbf-4

#train l3-xgbf
#train l3-nn



