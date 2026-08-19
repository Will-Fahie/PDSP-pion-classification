#!/usr/bin/env bash
# Promote one trained configuration to the unsuffixed filename that
# model-comparison, topology-selection and the analysis notebooks read.
#
# Usage: ./promote_model.sh <target> <tag>
#   ./promote_model.sh photon all8
#   ./promote_model.sh pion   base4

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <pion|photon> <tag>" >&2
    exit 1
fi

TARGET=$1
TAG=$2
PRED_SRC="predictions/predictions-mc/hybrid_cnn_mlp_${TAG}_${TARGET}.pkl"
PRED_DST="predictions/predictions-mc/hybrid_cnn_mlp_${TARGET}.pkl"
MODEL_SRC="models/hybrid_cnn_mlp_${TAG}_${TARGET}.pt"
MODEL_DST="models/hybrid_cnn_mlp_${TARGET}.pt"

for f in "$PRED_SRC" "$MODEL_SRC"; do
    if [ ! -f "$f" ]; then
        echo "missing: $f" >&2
        echo "has train_classifier.py been run for --model hybrid --target $TARGET --features $TAG ?" >&2
        exit 1
    fi
done

cp "$PRED_SRC" "$PRED_DST"
cp "$MODEL_SRC" "$MODEL_DST"

echo "promoted ${TAG} for ${TARGET}:"
echo "  $PRED_DST"
echo "  $MODEL_DST"
python3 -c "
import pickle
d = pickle.load(open('$PRED_DST', 'rb'))
print(f\"  {d['model_name']}  AUC={d['auc']:.3f}  purity={100*d['purity']:.1f}%  efficiency={100*d['efficiency']:.1f}%\")
"