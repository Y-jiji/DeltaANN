# DeltaANN

Data behind the evaluation figures of the paper "DeltaANN: Efficient In-place Insert & Delete in Large ANN Index for Steady Performance". Each file holds the values plotted in one figure. This repository does not include source code.

| Figure | Files | Values |
|---|---|---|
| 7 | `figure7/` | Throughput per batch and QPS at a fixed recall@10, 1:1:1 insert-delete-search, four 100M datasets |
| 8 | `figure8/` | The same for 1:1 insert-search, from 100M to 200M vectors |
| 9 | `figure9/` | OdinANN's time per batch, and SSD reads per delete, under 1:1:1 |
| 10 | `figure10/` | Hops and distance computations per insert under 1:1 |
| 11 | `figure11/` | Throughput per batch on SIFT from 900M vectors, under 1:1:1 and 1:1 |
| 12 | `figure12/` | Recall@10 at L = 100 under a distribution shift on SPACEV |
| 13 | `figure13/` | Throughput and SSD pages per replacement without each technique, SIFT100M, 1:1:1 |
| 14 | `figure14/` | Sensitivity to C, beta and tasks in flight per thread, SIFT100M, 1:1:1 |

## Notes

- A replacement is one delete and the insert that takes its place. Throughput counts inserts, deletes and searches.
- Figures 7 and 8: OdinANN and CONDA were stopped at or past the time DeltaANN took for all 100 batches, except OdinANN on SPACEV under 1:1, which finished.
- Figure 9(a): merge time comes from OdinANN's logs. The rest of each batch is the remainder.
- Figure 9(b): deletes have no read counters of their own, so these are estimates. CONDA's are the SSD reads not attributed to inserts and searches. DeltaANN's are the extra reads per insert under 1:1:1 over 1:1.
- Figure 10(b): PQ counts all PQ distances an insert computes, and exact counts exact distances.
- Figure 11: under 1:1:1, OdinANN was stopped after batch 54, having run 9.83 h against DeltaANN's 7.98 h, with merges at batches 14, 28 and 41. Under 1:1, it ran all 100 batches.
- Figure 14(b): re-prunes are the rate at batch 10, and 2/(C-1) is the amortized bound per insert or delete.
