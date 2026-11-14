import json
import pytest
from evaluation.audit_full_subset_parity import audit


def test_exact_subset_fields_order_and_missing_documents(tmp_path):
    rows = [dict(doc_id=str(i), title='', text=f'内容{i}') for i in range(4)]
    full = tmp_path / 'full'
    subset = tmp_path / 'subset'
    def write(path, values):
        path.write_text(''.join(json.dumps(r) + '\n' for r in values))
    write(full, rows)
    write(subset, [rows[0], rows[3]])
    assert audit(full, subset, tmp_path / 'ok', 4, 2)['exact_fields_and_relative_order']
    write(subset, [rows[3], rows[0]])
    with pytest.raises(ValueError, match='order mismatch'):
        audit(full, subset, tmp_path / 'order', 4, 2)
    write(subset, [rows[0], dict(rows[3], text='changed')])
    with pytest.raises(ValueError, match='document mismatch'):
        audit(full, subset, tmp_path / 'fields', 4, 2)
    write(subset, [rows[0], dict(rows[3], doc_id='absent')])
    with pytest.raises(ValueError, match='coverage'):
        audit(full, subset, tmp_path / 'missing', 4, 2)
