"""Compare every query, ordered candidate ID and score across two backends."""
import argparse,json,math
from pathlib import Path
from contracts.index_format import sha256_of
from evaluation.merge_full_captures import read_capture


def audit(reference,candidate,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    left,lp=read_capture(reference);right,rp=read_capture(candidate)
    for key in ('queries_sha256','documents','k','queries'):
        if lp[key]!=rp[key]:raise ValueError('protocol mismatch: '+key)
    byid={r['query_id']:r for r in right}
    if set(byid)!={r['query_id']for r in left}:raise ValueError('query universe mismatch')
    maximum=0.;hits=0
    for lrow in left:
        rrow=byid[lrow['query_id']]
        if lrow['text']!=rrow['text'] or lrow['explicit_constraints']!=rrow['explicit_constraints']:
            raise ValueError('query content mismatch')
        lhs=lrow['lexical'];rhs=rrow['lexical']
        if [(x['docid'],x['doc_id'])for x in lhs]!=[(x['docid'],x['doc_id'])for x in rhs]:
            raise ValueError('ordered ranking mismatch: '+lrow['query_id'])
        for a,b in zip(lhs,rhs):
            if not math.isclose(a['score'],b['score'],rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('score mismatch: '+lrow['query_id'])
            maximum=max(maximum,abs(a['score']-b['score']));hits+=1
    result={'status':'verified','queries':len(left),'documents':lp['documents'],
            'ordered_candidates_checked':hits,'maximum_absolute_score_difference':maximum,
            'reference_protocol_sha256':sha256_of(Path(reference)/'protocol.json'),
            'candidate_protocol_sha256':sha256_of(Path(candidate)/'protocol.json'),
            'reference_capture_sha256':lp['capture_sha256'],'candidate_capture_sha256':rp['capture_sha256'],
            'scope':'every declared query and ordered candidate; no latency equivalence claim'}
    with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ('reference','candidate','output'):p.add_argument('--'+n,required=True)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
