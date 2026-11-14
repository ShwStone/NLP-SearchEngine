# DuReader quality guard decision

The learned Router was trained only on the 172-query train split and evaluated on the frozen 1,020-query internal-dev split. It performed differently on the 100k subset and the full 8,096,668-document corpus:

- On 100k, learned Router nDCG@10 is `0.38770` at 60% heavy and 24 CE pairs on average. Independently executed fixed CE-24 is `0.38116`; Router minus fixed is `+0.00653`, grouped 95% CI `[-0.00849, 0.02180]`.
- On full corpus, learned Router nDCG@10 is `0.33638` at 83.0% heavy and 33.2 CE pairs on average. Independently executed fixed CE-33 is `0.35346`; Router minus fixed is `-0.01708`, grouped 95% CI `[-0.02450, -0.01022]`.

The full-corpus result is a quality regression. The repaired policy therefore activates a quality guard on the full corpus and routes every eligible query through fixed CE-33. This removes Router savings on that corpus, but preserves quality. The 100k learned policy remains research-only because its gain over fixed CE-24 is not statistically conclusive.

This is a repair and a negative result: the learned Router is not promoted as a universal default. The guard artifacts are `quality-guard.json` in each local experiment directory; the compact aggregate is `DUREADER_FULL_FINAL_REPORT.json`.

New guard receipts bind the independently scored control to the exact policy summary and development qrels. They record the development scope and set `production_default_safe` to false for either decision. The original experiment receipts in the aggregate retain their historical schema.
