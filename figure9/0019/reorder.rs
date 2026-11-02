use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::time::Instant;

use fnct::bench::DataStream;
use fnct::dataref::{CloneFrom, DRef, DataBox, DataMeta, DataPrimitive, Slice, SliceMeta};
use fnct::pt::{Point, PointMeta, PqConf, ProductQBuilder, l2sq_f32};

/// Points streamed from disk per read
const CHUNK: usize = 1 << 20;
/// Dimensions of the single band, every dimension zero-padded to a power of two
const BAND: usize = 128;
/// Code bits of the band, `2^BITS` centroids
const BITS: usize = 7;
/// Lloyd rounds, each a full scan of the dataset
const ROUNDS: usize = 3;
/// Seed of the centroid picks
const SEED: u64 = 16;
/// Contiguous blocks the seed sample is read in, spread evenly over the file
const BLOCKS: usize = 1000;
/// Points per seed sample block
const BLOCKLEN: usize = 1000;

/// Point element types the k-means runs over, which `ProductQBuilder` accepts
trait Element: DataPrimitive {
    fn asfloat(self) -> f32;
    fn seeded(sample: &DRef<Slice<Point<Self>>>) -> ProductQBuilder<Self>;
    fn iterate(builder: &mut ProductQBuilder<Self>, chunk: &DRef<Slice<Point<Self>>>);
}

/// `Element` for each element type of the datasets
macro_rules! element {
    ($($t:ty),*) => {$(
        impl Element for $t {
            fn asfloat(self) -> f32 {
                self as f32
            }
            fn seeded(sample: &DRef<Slice<Point<Self>>>) -> ProductQBuilder<Self> {
                ProductQBuilder::<$t>::seeded(BAND, BITS, SEED, sample)
            }
            fn iterate(builder: &mut ProductQBuilder<Self>, chunk: &DRef<Slice<Point<Self>>>) {
                builder.iterate(chunk);
            }
        }
    )*};
}

element!(i8, f32);

/// `BLOCKS` blocks of `BLOCKLEN` points spread evenly over the first `rows`
/// points of the headerless `path`
fn sample<T: Element>(path: &str, dims: usize, rows: usize) -> DataBox<Slice<Point<T>>> {
    let row = dims * size_of::<T>();
    let mut file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut bytes = vec![0u8; BLOCKS * BLOCKLEN * row];
    for (b, block) in bytes.chunks_exact_mut(BLOCKLEN * row).enumerate() {
        file.seek(SeekFrom::Start((b * (rows / BLOCKS) * row) as u64)).expect("seek sample");
        file.read_exact(block).expect("read sample");
    }
    let meta = SliceMeta::new(BLOCKS * BLOCKLEN, PointMeta::<T>::new(dims));
    let mut data = DataBox::<Slice<Point<T>>>::new(meta.clone());
    data.as_mut().clone_from(&meta.try_ref(&bytes));
    data
}

/// 128 centroids by `ROUNDS` Lloyd rounds, seeded from [`sample`], each
/// round streaming every point of `path` from disk through `iterate`
fn train<T: Element>(path: &str, dims: usize, rows: usize) -> PqConf {
    let seed = sample::<T>(path, dims, rows);
    let mut builder = T::seeded(seed.as_ref());
    drop(seed);
    for round in 0..ROUNDS {
        let clock = Instant::now();
        let mut stream = DataStream::<T>::new(path, dims, rows);
        while let Some(chunk) = stream.next(CHUNK, CHUNK) {
            T::iterate(&mut builder, chunk.as_ref());
        }
        builder.finish();
        eprintln!("reorder round={round} secs={:.1}", clock.elapsed().as_secs_f64());
    }
    builder.intoconf()
}

/// Index of the centroid of `book`, `BAND` floats each, nearest to `point`
/// by squared L2; the centroid entries past the point's dimension are zero
fn nearest(book: &[f32], point: &[f32]) -> u8 {
    let mut best = (f32::INFINITY, 0u8);
    for (i, centroid) in book.chunks_exact(BAND).enumerate() {
        let d = l2sq_f32(point, &centroid[..point.len()]);
        if d < best.0 {
            best = (d, i as u8);
        }
    }
    best.1
}

/// Nearest centroid of every point of `path`, each thread streaming its own
/// contiguous range from disk
fn assign<T: Element>(path: &str, dims: usize, rows: usize, conf: &PqConf) -> Vec<u8> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let span = rows.div_ceil(threads);
    let book = conf.book.as_slice();
    let mut labels = vec![0u8; rows];
    std::thread::scope(|scope| {
        for (t, part) in labels.chunks_mut(span).enumerate() {
            scope.spawn(move || {
                let lo = t * span;
                let mut stream = DataStream::<T>::new(path, dims, lo + part.len());
                stream.rewind(lo);
                let mut cast = vec![0f32; dims];
                let mut at = 0;
                while let Some(chunk) = stream.next(CHUNK, CHUNK) {
                    for i in 0..chunk.len() {
                        let point = chunk.get(i);
                        for (d, x) in cast.iter_mut().zip(point.data()) {
                            *d = (*x).asfloat();
                        }
                        part[at] = nearest(book, &cast);
                        at += 1;
                    }
                }
            });
        }
    });
    labels
}

/// Write the points of `path` into `out` grouped by `labels`, stable within
/// a cluster, scanning `path` once and appending through per-cluster buffers
/// `return`: the old id of the point at every new position
fn reorder<T: Element>(path: &str, dims: usize, labels: &[u8], out: &str) -> Vec<u32> {
    let row = dims * size_of::<T>();
    let mut cursor = vec![0usize; 1 << BITS];
    for &label in labels {
        cursor[label as usize] += 1;
    }
    let mut next = 0;
    for slot in cursor.iter_mut() {
        (*slot, next) = (next, next + *slot);
    }
    let file = File::create(out).unwrap_or_else(|e| panic!("create {out}: {e}"));
    file.set_len((labels.len() * row) as u64).expect("size reordered file");
    let mut reader = BufReader::with_capacity(1 << 24, File::open(path).expect("open base"));
    let mut buffers: Vec<Vec<u8>> = vec![Vec::new(); 1 << BITS];
    let mut perm = vec![0u32; labels.len()];
    let mut point = vec![0u8; row];
    for (old, &label) in labels.iter().enumerate() {
        let c = label as usize;
        reader.read_exact(&mut point).expect("read base");
        perm[cursor[c] + buffers[c].len() / row] = old as u32;
        buffers[c].extend_from_slice(&point);
        if buffers[c].len() >= 1 << 22 {
            file.write_all_at(&buffers[c], (cursor[c] * row) as u64).expect("write reordered");
            cursor[c] += buffers[c].len() / row;
            buffers[c].clear();
        }
    }
    for (c, buffer) in buffers.iter().enumerate() {
        file.write_all_at(buffer, (cursor[c] * row) as u64).expect("write reordered");
    }
    perm
}

/// `reorder <spacev|msturing> <base> <reordered> <perm> <centroids>`\
/// `perm` holds u32 old ids by new position, `centroids` 128 x `BAND` f32;
/// prints `cluster,size`
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (spacev, base, out) = (args[1] == "spacev", args[2].as_str(), args[3].as_str());
    assert!(spacev || args[1] == "msturing", "dataset {}, want spacev|msturing", args[1]);
    let (dims, size) = if spacev { (100, 1) } else { (100, 4) };
    let rows = std::fs::metadata(base).expect("stat base").len() as usize / (dims * size);
    let clock = Instant::now();
    let conf = if spacev { train::<i8>(base, dims, rows) } else { train::<f32>(base, dims, rows) };
    let labels = if spacev { assign::<i8>(base, dims, rows, &conf) } else { assign::<f32>(base, dims, rows, &conf) };
    eprintln!("reorder assigned rows={rows} secs={:.1}", clock.elapsed().as_secs_f64());
    let perm = if spacev { reorder::<i8>(base, dims, &labels, out) } else { reorder::<f32>(base, dims, &labels, out) };
    eprintln!("reorder written secs={:.1}", clock.elapsed().as_secs_f64());
    for (part, path) in [(perm.as_slice(), &args[4])] {
        let bytes = unsafe { std::slice::from_raw_parts(part.as_ptr() as *const u8, part.len() * 4) };
        let mut writer = BufWriter::with_capacity(1 << 24, File::create(path).expect("create perm"));
        writer.write_all(bytes).expect("write perm");
        writer.flush().expect("flush perm");
    }
    let book: Vec<u8> = conf.book.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(&args[5], book).expect("write centroids");
    let mut sizes = vec![0usize; 1 << BITS];
    for &label in &labels {
        sizes[label as usize] += 1;
    }
    println!("cluster,size");
    for (c, n) in sizes.iter().enumerate() {
        println!("{c},{n}");
    }
}
