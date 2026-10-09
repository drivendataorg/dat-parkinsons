#!/bin/bash
# Retrain Siamese exp 0 folds 0-4 with synthetic data, authors' hyperparameters, sequentially.
# Waits for the synthetic striatum prep to finish first.
cd /home/ec2-user/sfmn1/runs/siamese_synth
until grep -q "^rc=" /home/ec2-user/sfmn1/logs/step5_phase4_prep_synth.log; do sleep 20; done
grep -q "^rc=0" /home/ec2-user/sfmn1/logs/step5_phase4_prep_synth.log || { echo "prep failed"; exit 1; }
for f in 0 1 2 3 4; do
  echo "=== exp 0 fold $f start $(date +%T)"
  /usr/bin/time -f "elapsed %E" /home/ec2-user/sfmn1/code/.venv/bin/python -u /home/ec2-user/sfmn1/code/4_siamese_training/train_siamese.py --worker --exp 0 --fold $f --device cuda:0 > /home/ec2-user/sfmn1/logs/step6_siamese_synth_e0f$f.log 2>&1
  echo "=== exp 0 fold $f rc=$? end $(date +%T) $(tail -1 /home/ec2-user/sfmn1/logs/step6_siamese_synth_e0f$f.log)"
  cat results/siamese_network/tmp_preds/stats_exp_0_fold_$f.json; echo
done
echo ALLDONE
