mod command;

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::marker::PhantomData;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::time::{SystemTime, UNIX_EPOCH};

use command::{Command, command, knob};
use fnct::algo::pruner::{RNGPrune, RNGWaterConf};
use fnct::bench::{Bench, BenchConfig, DataStream, TagIndex};
use fnct::persist::AsyncPersist;
use fnct::dataref::{DRef, Data, DataBox, DataPrimitive, Slice, SliceMeta};
use fnct::db::{GraphVectorIndex, Profile, SSDGraphInstantConf, SSDGraphInstantIndex, SSDGraphInstantVertex, SSDInstantBufferConf, VectorIndex};
use fnct::io::{AsyncContext, AsyncFile, CoalescePool, LaioContext, LaioFile, StripeEvictor};
use fnct::pt::{L2, L2UpdatableQuantizer, Point, PointMeta, PqConf, ProductQ};

/// Points replaced per batch by default: the oldest removed, as many new
/// inserted and as many searched.
const STEP: usize = 1_000_000;
/// Points per insert-only call of the `fill` stage.
const FILL: usize = 100_000;
/// Sweeps after the load's, evenly spaced over the batches.
const CHECKPOINTS: usize = 10;
/// Out-neighbor slots of a vertex, the parlay graph's R, by default; knob
/// `degree` overrides, the flat graph's rows then that wide.
const DEGREE: usize = 64;
/// Neighbors an RNG prune keeps, half the slots so reverse edges append, by
/// default; knob `limit` overrides.
const LIMIT: usize = 32;
/// Squared RNG slack, parlay's alpha 1.2 squared.
const ALPHA: f64 = 1.44;
/// Beam of the search behind insert, parlay's L.
const CANDS: usize = 128;
/// β of watersearch's mask by default, knob `scale` overrides.
const SCALE: f64 = 1.0;
/// Hops the entry at the watermark must stay put before the watermark moves
/// past it, knob `stable` overrides.
const STABLE: usize = 4;
/// Watermark steps in a row keeping nothing before the slack is raised,
/// knob `stall` overrides.
const STALL: usize = 8;
/// Share of ids whose points train the codebook by default, knob `uprate`
/// overrides, 0 leaving it blank.
const UPRATE: f64 = 0.1;
/// Sampled base ids `retraincodebook` feeds the blank codebook at a time.
const RETRAIN: usize = 1_000_000;
/// Pages a search reads together in one round, PipeANN's and FreshDiskANN's
/// default beam width 4, so every system reads alike per hop.
const WIDTH: usize = 4;
/// Tasks in flight per thread, the fastest runmix of the GCP z3 runs
/// fnctntask-20261015-104014 and instntask-20261015-105742: one left the CPU
/// and the SSD idle, eight saturate the CPU.
const NTASK: usize = 8;
/// Bytes of one vertex page.
const PAGESZ: usize = 4096;
/// Kernel aio requests each bench worker's context holds in flight, so 128
/// workers stay inside the cluster's `fs.aio-max-nr` of 65536.
const IODEPTH: u32 = 256;
/// Pages the buffer pool holds by default, 80 MB, knob `memcap` overrides.
const MEMCAP: u64 = 20_000;
/// One of every `RATE` op rows is kept.
const RATE: usize = 50;
/// Beam sizes a sweep searches with, one QPS/recall point each.
const BEAMS: [usize; 10] = [10, 16, 24, 32, 50, 70, 100, 140, 200, 300];
/// Passes a sweep makes at each beam, QPS taken as their median.
const REPEAT: usize = 3;
/// Neighbors and beam of every search of a runmix.
const MIXK: usize = 50;
/// Knobs the driver takes, the first six shared by every mix driver;
/// `stable` and `stall` are watersearch's `stablehops` and `stallsteps`,
/// `scale` its β, `uprate` the codebook's update share, `blunt` true for the
/// blunt insert in its place, `book` a path the retrained codebook is written
/// to as `pqtrain` writes one; `tunes` (temporary) a list `8x8,8x4` of
/// threads by ntask, one grow batch each over the one loaded index; `open`
/// a snapshot to start from in place of the load and retrain, its search and
/// insert settings taken from the knobs; `save` a snapshot path written after
/// the batches; `start` the corpus row the batches insert from, `base` unless
/// a reopened snapshot already holds later rows; `band` the codebook's
/// dimensions per band, 8 by default; `degree` and `limit` the slots and the
/// prune's keep, `DEGREE` and `LIMIT` by default; `fill` the rows inserted
/// after the load, before the first sweep, 0 by default; `grow` the batches
/// that under mode=churn only insert before the removals start, 0 by default;
/// `frozen` true to never remove row 0, so the search entry id 0 keeps its
/// base point as a frozen start point, false by default.
const KEYS: [&str; 26] = [
    "batches", "every", "threads", "step", "qlen", "mode", "memcap", "wrthru", "width", "ntask",
    "stable", "stall", "scale", "uprate", "blunt", "book", "tunes", "open", "save", "start", "band",
    "degree", "limit", "fill", "grow", "frozen",
];
/// Counter names of `GraphProfile`, in column order; `wrote` counts the pages
/// an op's `multiwrite` calls dirtied, `mask` the pops watersearch screened
/// out, `adc` its ADC screen distances, `round1` to `round3` the ids kept at
/// slack 1, at alpha, and at every later level; `full` and `sdc` unused.
const NAMES: [&str; 15] = [
    "start", "laten", "hop", "dist", "estim", "cold", "pass", "mask", "full", "adc", "sdc", "wrote",
    "round1", "round2", "round3",
];

/// Vertex pool of points of `T` over an `O_DIRECT` file read and written by
/// per-thread kernel aio, a `multiwrite` relocating its records together onto
/// vacant or fresh pages, page `k` at byte `k` times the page size.
type Pool<T> = CoalescePool<LaioContext, StripeEvictor, SSDGraphInstantVertex<Point<T>>>;
/// Instant-deletion SSD graph over points of `T`, PQ coded, an insert's
/// edges kept by watersearch.
type Graph<T> = SSDGraphInstantIndex<ProductQ<T>, Pool<T>>;

/// Counters of one op by `NAMES`: wall-clock `start` and latency `laten` in
/// ns, then the index's events summed.
struct GraphProfile {
    counts: [u64; 15],
}

impl Profile for GraphProfile {
    fn new() -> Self {
        Self { counts: [0; 15] }
    }
    fn event(&mut self, name: &'static str, n: usize) {
        let clock = || SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
        match name {
            "start" => self.counts[0] = clock(),
            "end" => self.counts[1] = clock().saturating_sub(self.counts[0]),
            _ => {
                if let Some(at) = NAMES.iter().position(|&e| e == name) { self.counts[at] += n as u64; }
            }
        }
    }
    fn get(&self, name: &'static str) -> usize {
        NAMES.iter().position(|&e| e == name).map_or(0, |at| self.counts[at] as usize)
    }
    fn iter(&self) -> impl Iterator<Item = (&'static str, usize)> {
        NAMES.iter().zip(self.counts).map(|(&name, n)| (name, n as usize))
    }
}

/// Load `base` vertices into the empty `db`, as `Bench::vectorgraphload`
/// does: points from the corpus's first rows, out-neighbors from the flat
/// graph's rows of `degree` u32, `u32::MAX` an empty slot\
/// SHAME(MANYARG)
fn load<T: DataPrimitive>(db: Graph<T>, base: usize, dim: usize, degree: usize, graph: &File, points: &File) -> Graph<T>
where
    ProductQ<T>: L2UpdatableQuantizer<Pt = Point<T>, Conf = PqConf, DistSq = <Point<T> as L2>::DistSq> + Sync,
    Point<T>: L2 + Data<Meta = PointMeta<T>> + Sync,
    Graph<T>: GraphVectorIndex<P = Point<T>>,
{
    let width = DataBox::<Point<T>>::new(PointMeta::<T>::new(dim)).bytes().len();
    db.graphload(base, &|range: Range<usize>| {
        let mut rows = DataBox::<Slice<Point<T>>>::new(SliceMeta::new(range.len(), PointMeta::<T>::new(dim)));
        points.read_exact_at(unsafe { rows.bytes_mut() }, (range.start * width) as u64).expect("read vector rows");
        let mut raw = vec![0u8; range.len() * degree * 4];
        graph.read_exact_at(&mut raw, (range.start * degree * 4) as u64).expect("read graph rows");
        let mut edges = DataBox::<Slice<usize>>::new(SliceMeta::new(range.len() * degree, PhantomData));
        let slots: &mut [usize] = edges.as_mut().as_mut();
        for (slot, b) in slots.iter_mut().zip(raw.chunks_exact(4)) {
            let e = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            *slot = if e == u32::MAX { usize::MAX } else { e as usize };
        }
        (rows, edges)
    })
}

/// Write the centroids of `cbook` to `path` as `pqtrain` writes a codebook:
/// raw little-endian f32, band after band, taken from the codebook's
/// serialized `book`, the first of its fields
fn dumpbook<T>(cbook: &ProductQ<T>, path: &str) {
    let value = serde_json::to_value(cbook).unwrap_or_else(|e| panic!("serialize codebook: {e}"));
    let book: Vec<f32> = serde_json::from_value(value[0].clone()).unwrap_or_else(|e| panic!("codebook book: {e}"));
    let bytes: Vec<u8> = book.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

/// Search `qs` `REPEAT` times at every beam of `BEAMS`, the result list as
/// wide as the beam, tagged `sweep_b{batch}_k{k}_r{r}`\
/// SHAME(WIDEFUNC)
fn sweep<DB, T: DataPrimitive>(bench: &mut Bench<LaioContext, DB, GraphProfile, T>, qs: &DRef<'_, Slice<Point<T>>>, batch: usize)
where
    DB: VectorIndex<P = Point<T>, Id = u32>,
    DB::D: Clone + PartialOrd + Copy + Default + Send + Into<f64>,
{
    for k in BEAMS {
        for r in 0..REPEAT {
            bench.runverify(format!("sweep_b{batch}_k{k}_r{r}"), qs, k);
        }
    }
}

/// Load the flat graph over the corpus's first rows into a fresh instant SSD
/// graph on the index path, its blank codebook then retrained from the
/// sampled base ids, insert `fill` rows `FILL` at a time, then run `batches`
/// runmix calls, batch b tagged `mix_{b}` removing rows (b - grow) * step to
/// (b - grow + 1) * step, or `grow_{b}` under mode=grow or before `grow`,
/// inserting the next step rows and searching as many, sweeping the queries
/// after the load, every `every` batches and after the last\
/// SHAME(TALLFUNC) SHAME(WIDEFUNC)
fn churn<T: DataPrimitive>(dim: usize, qlen: usize, cmd: &Command)
where
    ProductQ<T>: L2UpdatableQuantizer<Pt = Point<T>, Conf = PqConf, DistSq = <Point<T> as L2>::DistSq> + Sync,
    Point<T>: L2 + Data<Meta = PointMeta<T>> + Sync,
    <Point<T> as L2>::DistSq: Copy + Default + PartialOrd + Into<f64> + Send,
    Graph<T>: GraphVectorIndex + VectorIndex<P = Point<T>, Id = u32, D = <Point<T> as L2>::DistSq>,
{
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let nthread: usize = knob(cmd, "threads", cores);
    let step: usize = knob(cmd, "step", STEP);
    let qlen: usize = knob(cmd, "qlen", qlen);
    let mode: String = knob(cmd, "mode", "churn".to_string());
    let memcap: u64 = knob(cmd, "memcap", MEMCAP);
    let wrthru: bool = knob(cmd, "wrthru", false);
    let width: usize = knob(cmd, "width", WIDTH);
    let ntask: usize = knob(cmd, "ntask", NTASK);
    let scale: f64 = knob(cmd, "scale", SCALE);
    let stable: usize = knob(cmd, "stable", STABLE);
    let stall: usize = knob(cmd, "stall", STALL);
    let uprate: f64 = knob(cmd, "uprate", UPRATE);
    let blunt: bool = knob(cmd, "blunt", false);
    let book: String = knob(cmd, "book", String::new());
    let tunes = tunelist(&knob(cmd, "tunes", String::new()));
    let open: String = knob(cmd, "open", String::new());
    let save: String = knob(cmd, "save", String::new());
    let degree: usize = knob(cmd, "degree", DEGREE);
    let limit: usize = knob(cmd, "limit", LIMIT);
    let fill: usize = knob(cmd, "fill", 0);
    let grow: usize = knob(cmd, "grow", 0);
    let frozen: bool = knob(cmd, "frozen", false);
    let concur = tunes.iter().map(|&(t, n)| t * n.max(4)).max().unwrap_or(nthread * ntask.max(4));
    assert!(mode == "churn" || mode == "grow", "mode={mode}, want churn|grow");
    assert!(nthread > 0 && step > 0, "threads and step must be positive");
    let size = |path: &str| std::fs::metadata(path).unwrap_or_else(|e| panic!("stat {path}: {e}")).len() as usize;
    let base = size(&cmd.graph) / (degree * std::mem::size_of::<u32>());
    let rows = size(&cmd.point) / (dim * std::mem::size_of::<T>());
    let start: usize = knob(cmd, "start", base);
    let spare = rows.saturating_sub(start + fill) / step;
    let batches: usize = knob(cmd, "batches", (base / step).min(spare.saturating_sub(1)));
    let every: usize = knob(cmd, "every", (batches / CHECKPOINTS).max(1));
    assert!(every > 0 && (base + fill) % step == 0,
        "base {base} plus fill {fill} must be a multiple of step, every positive");
    let reads = start + fill + (batches + 1) * step;
    assert!(reads <= rows, "the corpus holds {rows} rows, a run of {batches} batches reads {reads}");
    eprintln!("instantmix mode={mode} dim={dim} base={base} fill={fill} grow={grow} frozen={frozen} step={step} batches={batches} every={every}");
    eprintln!("instantmix uprate={uprate} blunt={blunt} stable={stable} stall={stall} scale={scale}");
    eprintln!("instantmix memcap={memcap} wrthru={wrthru} pagesz={PAGESZ} threads={nthread} repeat={REPEAT}");
    eprintln!("instantmix degree={degree} limit={limit} alpha={ALPHA} cands={CANDS} width={width} ntask={ntask} mixk={MIXK} rate={RATE}");
    let band: usize = knob(cmd, "band", 8);
    let shape = PqConf { dims: dim, band, bits: 8 };
    let mut centr = DataBox::<Point<T>>::new(PointMeta::<T>::new(dim));
    unsafe { centr.bytes_mut().fill(0) };
    let conf = SSDGraphInstantConf {
        buffersetup: SSDInstantBufferConf { memcap, concur: concur as u64, pagesz: PAGESZ, wrthru },
        searchwidth: width,
        insertcands: CANDS,
        graphdegree: degree,
        screenscale: scale,
        quantuprate: uprate,
        insertsetup: RNGWaterConf { stablehops: stable, stallsteps: stall },
        bluntinsert: blunt,
    };
    let prune = RNGPrune { alpha: ALPHA, limit };
    let mut ctx = LaioContext::new(1, IODEPTH);
    let mut stream = DataStream::<T>::new(&cmd.point, dim, rows);
    stream.rewind(start);
    let qbox = DataStream::<T>::new(&cmd.query, dim, qlen).next(qlen, qlen).expect("read queries");
    let qs = qbox.as_ref();
    let config = BenchConfig { nthread, ntask, iodepth: IODEPTH, profilerate: RATE };
    let graph = File::open(&cmd.graph).unwrap_or_else(|e| panic!("open {}: {e}", cmd.graph));
    let points = File::open(&cmd.point).unwrap_or_else(|e| panic!("open {}: {e}", cmd.point));
    let mut bench = ctx.block(async {
        let tags = if open.is_empty() {
            let file = LaioFile::create(&cmd.index).await.expect("create vertex file");
            let db = Graph::<T>::new(file, conf, shape, &centr.as_ref(), prune).await;
            eprintln!("instantmix load start");
            let db = load(db, base, dim, degree, &graph, &points);
            eprintln!("instantmix retrain start");
            let db = db.retraincodebook(RETRAIN).await;
            let db = if book.is_empty() { db } else {
                let (db, cbook) = unsafe { db.exportcodebook() }.await;
                dumpbook(&cbook, &book);
                eprintln!("instantmix codebook written to {book}");
                db
            };
            TagIndex::new(db, (0..base).map(|i| (i as u64, i as u32)))
        } else {
            let snap = File::open(&open).unwrap_or_else(|e| panic!("open {open}: {e}"));
            let mut tags = TagIndex::<Graph<T>>::load(BufReader::new(snap)).await.expect("load snapshot");
            tags.engine().reconf(&conf);
            eprintln!("instantmix opened {open}, rows from {start}");
            tags
        };
        Bench::<LaioContext, _, GraphProfile, T>::wrap(tags, stream, config, &cmd.trace)
    });
    eprintln!("instantmix load done");
    for k in 0..fill.div_ceil(FILL) {
        let n = FILL.min(fill - k * FILL);
        bench.runinsert(format!("fill_{k}"), n);
        eprintln!("instantmix fill {} done, rows {} to {} in", k + 1, start + k * FILL, start + k * FILL + n);
    }
    sweep(&mut bench, &qs, 0);
    if !tunes.is_empty() {
        let restream = |at: usize| {
            let mut s = DataStream::<T>::new(&cmd.point, dim, rows);
            s.rewind(at);
            s
        };
        tune(bench, &tunes, &restream, base, step, &cmd.trace);
        return;
    }
    for b in 0..batches {
        let remove: Vec<u64> = match mode.as_str() {
            "churn" if b >= grow => (((b - grow) * step) as u64..((b - grow + 1) * step) as u64)
                .filter(|&row| !(frozen && row == 0)).collect(),
            _ => Vec::new(),
        };
        let tag = if remove.is_empty() { "grow" } else { "mix" };
        bench.runmix(format!("{tag}_{b}"), step, step, MIXK, &remove);
        eprintln!("instantmix batch {} done, {} rows out from {}, rows {} to {} in", b + 1, remove.len(),
            b.saturating_sub(grow) * step,
            start + fill + b * step, start + fill + (b + 1) * step);
        if (b + 1) % every == 0 || b + 1 == batches { sweep(&mut bench, &qs, b + 1); }
    }
    if !save.is_empty() {
        let tags = bench.unwrap();
        let mut out = BufWriter::new(File::create(&save).unwrap_or_else(|e| panic!("create {save}: {e}")));
        ctx.block(async { tags.dump(&mut out).await.expect("save snapshot") });
        out.flush().expect("flush snapshot");
        eprintln!("instantmix saved {save}, rows through {}", start + fill + batches * step);
    }
}

/// Parse a `tunes` text `8x8,8x4` into (threads, ntask) pairs, none for an
/// empty text
fn tunelist(text: &str) -> Vec<(usize, usize)> {
    text.split(',').filter(|s| !s.is_empty()).map(|s| {
        let (t, n) = s.split_once('x').unwrap_or_else(|| panic!("tune {s}, want <threads>x<ntask>"));
        (t.parse().expect("tune threads"), n.parse().expect("tune ntask"))
    }).collect()
}

/// Temporary (#015, z3 bottlenecks): one `step` grow batch per (threads,
/// ntask) of `list` over the one loaded index, the bench re-wrapped with that
/// config and a stream rewound to the next unused row, tagged
/// `tune_{t}x{n}`, its log at `{trace}.{t}x{n}`\
/// SHAME(WIDEFUNC)
fn tune<T: DataPrimitive>(bench: Bench<LaioContext, Graph<T>, GraphProfile, T>, list: &[(usize, usize)], restream: &dyn Fn(usize) -> DataStream<T>, base: usize, step: usize, trace: &str)
where
    ProductQ<T>: L2UpdatableQuantizer<Pt = Point<T>, Conf = PqConf, DistSq = <Point<T> as L2>::DistSq> + Sync,
    Point<T>: L2 + Data<Meta = PointMeta<T>> + Sync,
    <Point<T> as L2>::DistSq: Copy + Default + PartialOrd + Into<f64> + Send,
    Graph<T>: GraphVectorIndex + VectorIndex<P = Point<T>, Id = u32, D = <Point<T> as L2>::DistSq>,
{
    let mut bench = bench;
    for (i, &(nthread, ntask)) in list.iter().enumerate() {
        let tags = bench.unwrap();
        let config = BenchConfig { nthread, ntask, iodepth: IODEPTH, profilerate: RATE };
        bench = Bench::wrap(tags, restream(base + i * step), config, format!("{trace}.{nthread}x{ntask}"));
        bench.runmix(format!("tune_{nthread}x{ntask}"), step, step, MIXK, &[]);
        eprintln!("instantmix tune {nthread}x{ntask} done");
    }
}

/// Dispatch on the dataset: sift is 128 u8 values a point with 10K queries,
/// msturing 100 f32 with 10K, spacev 100 i8 with 29316, deep 96 f32 with 10K.
fn main() {
    let cmd = command(&KEYS);
    match cmd.label.as_str() {
        "sift" => churn::<u8>(128, 10_000, &cmd),
        "msturing" => churn::<f32>(100, 10_000, &cmd),
        "spacev" => churn::<i8>(100, 29_316, &cmd),
        "deep" => churn::<f32>(96, 10_000, &cmd),
        other => panic!("dataset {other}, want sift|msturing|spacev|deep"),
    }
}
