"""Predict raw CE depth for a batch of queries from first-pass BM25 documents."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import expit
from torch import nn
from torch.nn import functional as F

from .ce_factor_router import route_batch


class Adapter(nn.Module):
    """Learned query-document similarity on frozen 512-dimensional BGE vectors."""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(512, 32, bias=False)
        self.document = nn.Linear(512, 32, bias=False)
        self.query_bias = nn.Linear(512, 1)
        self.document_bias = nn.Linear(512, 1)

    def forward(self, query, documents):
        query, documents = query.float(), documents.float()
        cosine = (query[:, None, :] * documents).sum(-1)
        left = F.normalize(self.query(query), dim=-1)
        right = F.normalize(self.document(documents), dim=-1)
        score = 12 * (cosine - .55) + 5 * (left[:, None, :] * right).sum(-1)
        return score + self.query_bias(query) + self.document_bias(documents).squeeze(-1)


class Candidates:
    """Legal inputs in BM25 order. Index zero in the vector table is padding."""

    def __init__(self, data):
        self.ids = data['ids']
        self.query = data['q']
        self.local = data['local']
        self.sim = data['sim']
        self.mask = data['mask']
        self.index = data['ind']
        self.table = data['table']
        self.vec = data['v']
        n = len(self.ids)
        if self.mask.shape != (n, 200) or self.mask.dtype != np.bool_:
            raise ValueError('mask must be a boolean matrix with 200 positions per query')
        if self.local.shape != (n, 200, 9) or self.query.shape != (n, 96):
            raise ValueError('expected nine document fields and 96 query fields')
        if self.sim.shape != self.mask.shape or self.index.shape != self.mask.shape:
            raise ValueError('document arrays must have the same number of ranks')
        if self.vec.shape != (n, 512) or self.table.ndim != 2 or self.table.shape[1] != 512:
            raise ValueError('query and document vectors must have 512 dimensions')
        if self.index.dtype.kind not in 'iu' or np.any(self.index < 0) or np.any(self.index >= len(self.table)):
            raise ValueError('invalid indices into the document vector table')
        if not np.array_equal(self.index != 0, self.mask):
            raise ValueError('padding and document mask disagree')
        order = np.argsort(-np.where(self.mask, self.sim, -np.inf), axis=1, kind='stable')
        self.position = np.argsort(order, axis=1).astype(np.int16) + 1
        self.gap = (np.take_along_axis(self.sim, order[:, :1], axis=1) - self.sim).astype(np.float32)

    def membership_fields(self, rows, ranks):
        position = self.position[rows, ranks].astype(np.float32)
        return np.column_stack([
            self.local[rows, ranks].astype(np.float32), np.log1p(ranks), ranks / 200,
            self.query[rows, :8], self.sim[rows, ranks], self.gap[rows, ranks],
            np.log1p(position), position - (1 + ranks), 1 / (60 + position) + 1 / (61 + ranks)
        ]).astype(np.float32)

    def document_fields(self, rows, ranks):
        return np.column_stack([
            self.local[rows, ranks], np.log1p(ranks), self.sim[rows, ranks],
            np.log1p(self.position[rows, ranks]), self.query[rows, :8],
            self.relevance[rows, ranks], self.membership[rows, ranks]
        ]).astype(np.float32)

    def nominate(self, model, adapter, device):
        self.relevance = np.zeros(self.mask.shape, np.float32)
        self.membership = np.zeros_like(self.relevance)
        intercept, slope = model['platt']
        queries = torch.from_numpy(self.vec).to(device)
        # Transfer only the vectors needed by the current batch.
        with torch.inference_mode():
            for start in range(0, len(self.ids), 96):
                rows = np.arange(start, min(start + 96, len(self.ids)))
                docs = torch.from_numpy(self.table[self.index[rows]]).to(device)
                with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                    raw = adapter(queries[rows], docs)
                prob = expit(intercept + slope * raw.float().cpu().numpy())
                self.relevance[rows] = np.where(self.mask[rows], prob, 0)
                rr = np.repeat(rows, 200)
                fields = self.membership_fields(rr, np.tile(np.arange(200), len(rows)))
                membership = model['membership'].predict_proba(fields)[:, 1].reshape(-1, 200)
                self.membership[rows] = np.where(self.mask[rows], membership, 0)
        self.nominees = np.full((len(self.ids), 12), -1, np.int16)
        for i, mask in enumerate(self.mask):
            valid = np.flatnonzero(mask)
            p, m = self.relevance[i], self.membership[i]
            left = valid[np.argsort(-(p * m)[valid], kind='stable')[:8]]
            rest = np.setdiff1d(valid, left)
            right = rest[np.argsort(-(p * (1 - m))[rest], kind='stable')[:4]]
            self.nominees[i, :len(left)] = left
            self.nominees[i, 8:8 + len(right)] = right


def pair_features(batch, estimator):
    n = len(batch.ids)
    nom = batch.nominees
    rank = nom.clip(min=0)
    rows = np.arange(n)[:, None]
    p, m = batch.relevance[rows, rank], batch.membership[rows, rank]
    weights = np.zeros((n, 8, 4))
    present = (nom[:, :8, None] >= 0) & (nom[:, None, 8:] >= 0)
    i, a, b = np.nonzero(present)
    for start in range(0, len(i), 2048):
        s = slice(start, start + 2048)
        q, left, right = i[s], nom[i[s], a[s]], nom[i[s], 8 + b[s]]
        x = batch.document_fields(q, left)
        y = batch.document_fields(q, right)
        v = batch.table[batch.index[q, left]].astype(np.float32)
        u = batch.table[batch.index[q, right]].astype(np.float32)
        columns = [1, 2, 3, 4, 9, 10, 11, -2, -1]
        design = np.column_stack([x, y, x[:, columns] - y[:, columns], np.einsum('ij,ij->i', v, u)])
        weights[q, a[s], b[s]] = np.clip(estimator.predict(design), 0, 1)
    weights *= m[:, :8, None]
    weights *= 1 - m[:, None, 8:]
    gain = (weights * p[:, :8, None]).sum(2) / np.maximum((nom[:, 8:] >= 0).sum(1), 1)[:, None]
    loss = (weights * p[:, None, 8:]).sum(1) / np.maximum((nom[:, :8] >= 0).sum(1), 1)[:, None]
    return np.column_stack([gain.sum(1), gain.max(1), loss.sum(1), loss.max(1)]).astype(np.float32)


def prior_fields(batch, pair):
    joint = batch.relevance * batch.membership
    bands = [(0, 10), (10, 28), (28, 40), (40, 60), (60, 100), (100, 200)]
    summaries = [fn(joint[:, a:b], axis=1) for a, b in bands for fn in (np.max, np.sum)]
    summaries.extend(fn(batch.relevance[:, a:b], axis=1) for a, b in bands[:3] for fn in (np.max, np.sum))
    summaries.append(joint.sum(1) / (1 + batch.relevance[:, :28].sum(1)))
    return np.column_stack([batch.query, *summaries, pair])


def design(batch, pair, challenger):
    nom = batch.nominees
    ranks = nom[:, :8] if challenger else nom[:, 8:]
    n, width = ranks.shape
    context = pair
    if not challenger:
        left = nom[:, :8].clip(min=0)
        rows = np.arange(n)[:, None]
        chance = np.where(nom[:, :8] >= 0, batch.relevance[rows, left] * batch.membership[rows, left], 0)
        context = np.column_stack([pair, chance.max(1), chance.sum(1) / np.maximum((nom[:, :8] >= 0).sum(1), 1)])
    fields = batch.document_fields(np.repeat(np.arange(n), width), ranks.clip(min=0).ravel())
    x = np.column_stack([fields, np.tile(np.arange(width), n), np.repeat(context, width, axis=0)])
    return x.astype(np.float32), ranks >= 0


def predict(data, model, adapter, slots, device):
    batch = Candidates(data)
    if not 0 <= slots <= len(batch.ids) or not len(batch.ids):
        raise ValueError('expected a nonempty batch with a feasible upgrade quota')
    batch.nominate(model, adapter.eval().to(device), device)
    pair = pair_features(batch, model['joint'])
    prior = model['router'].predict(prior_fields(batch, pair))
    depths = np.full(len(prior), 28)
    depths[np.lexsort((batch.ids, -prior))[:slots]] = 40
    x, active = design(batch, pair, True)
    xx = x[active.ravel()]
    relevance, place, event = model['factor']
    rel = np.zeros(active.shape)
    bands = np.zeros((*active.shape, 3))
    events = np.zeros_like(bands)
    amount = np.zeros_like(rel)
    if len(xx):
        rel[active] = relevance.predict_proba(xx)[:, 1]
        bands[active] = place.predict_proba(xx)
        for k in (1, 2):
            events[:, :, k][active] = event.predict_proba(np.column_stack([xx, np.full(len(xx), k)]))[:, 1]
        amount[active] = np.clip(model['amount'].predict(xx), 0, 1)
    x, valid = design(batch, pair, False)
    losses = np.zeros((*valid.shape, 2))
    if valid.any():
        for col, head in enumerate(model['loss']):
            losses[:, :, col][valid] = np.clip(head.predict(x[valid.ravel()]), 0, 1)
    calibration = dict(net_and_losses=model['coef'], challenger_gain=model['extra'])
    return route_batch(batch.ids, prior, depths, rel, bands, events, amount, losses, calibration)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--inputs', type=Path, required=True)
    cli.add_argument('--model-dir', type=Path, required=True)
    cli.add_argument('--slots', type=int, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--device', default='cuda')
    args = cli.parse_args()
    model = joblib.load(args.model_dir / 'heads.joblib')
    adapter = Adapter()
    adapter.load_state_dict(torch.load(args.model_dir / 'adapter.pt', map_location='cpu', weights_only=True))
    with np.load(args.inputs, allow_pickle=False) as data:
        actions, _, _ = predict(data, model, adapter, args.slots, torch.device(args.device))
        with args.output.open('xb') as fp:
            np.savez_compressed(fp, ids=data['ids'], depth=actions)


if __name__ == '__main__':
    main()
