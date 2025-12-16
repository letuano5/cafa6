import pandas as pd
import os



pp_min = os.path.join('./protlib/scripts/postproc', f'pred_min.tsv')
pp_max = os.path.join('./protlib/scripts/postproc', f'pred_max.tsv')

pred_max = pd.read_csv(pp_max, sep="\t", header=None,
                       names=["EntryID", "term", "prob"],
                       dtype={"EntryID": str, "term": str, "prob": float})

pred_min = pd.read_csv(pp_min, sep="\t", header=None,
                       names=["EntryID", "term", "prob"],
                       dtype={"EntryID": str, "term": str, "prob": float})

pred = (pd.merge(pred_max, pred_min,
                 on=["EntryID", "term"],
                 how="outer")
          .fillna(0)
          .reset_index(drop=True))

max_rate=0.5
pred['prob'] = pred['prob_x'] * max_rate + pred['prob_y'] * (1 - max_rate)
pred = pred[['EntryID', 'term', 'prob']]

output_path='submission.tsv'
pred["prob"] = pred["prob"].map(lambda x: f"{x:.3f}")
pred.to_csv(output_path, sep="\t", header=False, index=False)
