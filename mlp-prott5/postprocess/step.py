import argparse
import os
import sys

import tqdm
import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str, required=True)

parser.add_argument('-d', '--device', type=str, default="cpu")  # kept for interface compatibility
parser.add_argument('-b', '--batch_size', type=int, default=20000)
parser.add_argument('-bi', '--batch_inner', type=int, default=5000)
parser.add_argument('-l', '--lr', type=float, default=0.1)
parser.add_argument('-dr', '--direction', type=str, default='max')


def propagate_max(mat, G):
    # mat: numpy array shape (batch_len, G.idxs)
    # G.order assumed ordering of nodes
    for f in G.order:
        adj = G.terms_list[f].get('children', [])
        if len(adj) == 0:
            continue
        adj = np.asarray(adj, dtype=np.int64)
        # compute max over adj columns for every row, then combine
        # new_val = max(current_val, max(mat[:, adj]))
        max_from_children = mat[:, adj].max(axis=1)
        mat[:, f] = np.maximum(mat[:, f], max_from_children)
    return


def propagate_min(mat, G):
    """
    D expected as dict: depth (int) -> list of node indices.
    For each depth (in increasing order) update mat[:, f] = min(mat[:, f], min(mat[:, adj], axis=1))
    """
    D = get_depths(G, True)  # returns dict: depth -> [node_indices]

    for depth in sorted(D.keys()):
        for f in D[depth]:
            adj = G.terms_list[f].get('adj', [])
            if not adj:
                continue
            adj_arr = np.asarray(adj, dtype=np.int64)
            # compute min across adj columns per-row
            min_from_adj = np.min(mat[:, adj_arr], axis=1)
            mat[:, f] = np.minimum(mat[:, f], min_from_adj)
    return


if __name__ == '__main__':
    # for dirname, _, filenames in os.walk('./'):
    #     for filename in filenames:
    #         print(os.path.join(dirname, filename))
    # exit(0)
    args = parser.parse_args()

    # CPU-only run: ignore CUDA_* env vars
    import numpy as np
    import pandas as pd

    try:
        from metric import get_funcs_mapper, get_ns_id, obo_parser, Graph, get_depths
    except Exception:
        get_funcs_mapper, get_ns_id, obo_parser, Graph, get_depths = [None] * 5

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    graph_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
    pp_path = os.path.join(config['base_path'], "protlib/scripts", 'postproc')

    # print(graph_path, pp_path)
    input_path = os.path.join(pp_path, 'pred.tsv')
    output_path = os.path.join(pp_path, f'pred_{args.direction}.tsv')

    # read with pandas
    trainTerms = pd.read_csv(input_path, sep='\t', names=['EntryID', 'term', 'prob'], header=None, dtype={'EntryID': object, 'term': object, 'prob': float})

    # build ontologies list
    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    # build mapping EntryID -> integer id (0..n-1)
    back_prot_id = trainTerms['EntryID'].drop_duplicates().reset_index(drop=True)
    length = len(back_prot_id)
    prot_id_map = pd.Series(range(length), index=back_prot_id.values)
    # map EntryID to id (int)
    trainTerms = trainTerms.merge(prot_id_map.rename('id'), left_on='EntryID', right_index=True, how='left')

    # flag for first write
    flg = True

    for i in tqdm.tqdm(range(0, length, args.batch_size)):
        # sample rows whose id in [i, i+batch_size)
        batch_mask = (trainTerms['id'] >= i) & (trainTerms['id'] < i + args.batch_size)
        sample = trainTerms.loc[batch_mask].copy()
        batch_len = min(args.batch_size, length - i)

        if sample.shape[0] == 0:
            # nothing in this batch
            continue

        for G in ontologies:
            # mapper: term -> col index
            mapper = pd.Series(get_funcs_mapper(G))
            # map term -> term_id; terms not found become NaN
            sample['term_id'] = sample['term'].map(mapper)
            sample_ont = sample.dropna(subset=['term_id']).copy()
            # ensure integer dtype
            sample_ont['term_id'] = sample_ont['term_id'].astype(np.int32)

            # shift id to batch-local indices (0..batch_len-1)
            sample_ont['id_local'] = sample_ont['id'] - i
            sample_ont['id_local'] = sample_ont['id_local'].astype(np.int32)

            # build matrix: rows = batch_len, cols = G.idxs
            mat = np.zeros((batch_len, G.idxs), dtype=np.float32)

            # scatter_add: add probs to mat[id_local, term_id]
            # handle duplicates using numpy.add.at
            rows = sample_ont['id_local'].values
            cols = sample_ont['term_id'].values
            vals = sample_ont['prob'].values.astype(np.float32)
            np.add.at(mat, (rows, cols), vals)

            # clip and copy old
            np.clip(mat, 0.0, 1.0, out=mat)
            mat_old = mat.copy()

            # propagate
            if args.direction == 'max':
                propagate_max(mat, G)
            else:
                propagate_min(mat, G)

            # smoothing
            mat = mat * args.lr + mat_old * (1.0 - args.lr)

            # output per inner-batch to avoid huge intermediate DataFrame
            for j in range(0, mat.shape[0], args.batch_inner):
                mat_batch = mat[j: j + args.batch_inner]
                row_idx, col_idx = np.nonzero(mat_batch)

                if len(row_idx) == 0:
                    continue

                # build DataFrame
                entry_ids = back_prot_id.iloc[i + j + row_idx].reset_index(drop=True)
                # map col_idx -> term string (reverse mapper)
                inv_map = pd.Series(get_funcs_mapper(G, False))
                terms = inv_map.iloc[col_idx].reset_index(drop=True)

                probs = mat_batch[row_idx, col_idx]
                sample_batch = pd.DataFrame({
                    'EntryID': entry_ids.values,
                    'term': terms.values,
                    'prob': probs
                })

                # sort as original
                sample_batch = sample_batch.sort_values(['EntryID', 'term'], ascending=[True, True]).reset_index(drop=True)

                # format prob similar to original (short string)
                # Here we use 4 decimal places and then slice to 5 chars to mimic original behavior
                sample_batch['prob'] = sample_batch['prob'].map(lambda x: f"{x:.4f}").astype(str).str.slice(0, 5)

                ns_id, ns_str = get_ns_id(G)
                asp = ns_str.upper() + 'O'

                # write to file (w for first, append otherwise)
                mode = 'w' if flg else 'a'
                with open(output_path, mode) as f:
                    sample_batch.to_csv(f, index=False, sep='\t', header=None)
                flg = False

    print("Done. Output written to:", output_path)
