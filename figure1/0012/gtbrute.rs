use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::pin::pin;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicU32, AtomicUsize};
use std::task::{Context, Poll, Waker};

use fnct::dataref::{Data, DataPrimitive};
use fnct::db::{BruteConfig, BruteDB, pref};
use fnct::io::TokioContext;
use fnct::pt::{L2, Point, PointMeta};

/// Neighbors kept per query, the depth of the ground truth file
const TOPK: usize = 10;
/// Points buffered before a flush offers them to every query
const BUFLEN: usize = 1 << 16;

/// Row-major points read from a ParlayANN bin file
struct Matrix<T> {
    rows: usize,
    dims: usize,
    data: Vec<T>,
}

/// Read a ParlayANN bin file: `u32 rows, u32 dims`, then rows x dims elements
fn load<T: DataPrimitive>(path: &str) -> Matrix<T> {
    let mut reader = BufReader::with_capacity(1 << 24, File::open(path).unwrap());
    let mut head = [0u8; 8];
    reader.read_exact(&mut head).unwrap();
    let rows = u32::from_le_bytes(head[0..4].try_into().unwrap()) as usize;
    let dims = u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize;
    let mut data = vec![T::default(); rows * dims];
    let size = data.len() * size_of::<T>();
    let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, size) };
    reader.read_exact(bytes).unwrap();
    let mut rest = [0u8; 1];
    assert_eq!(reader.read(&mut rest).unwrap(), 0, "{path}: trailing bytes after {rows} x {dims}");
    Matrix { rows, dims, data }
}

/// Exact top-`TOPK` of every query over every base point, ids being base rows
///
/// Queries are registered on the empty index, so each flush offers its
/// points to all of them and the base is read once. Threads insert rows
/// in parallel and the index numbers inserts by arrival, so `rows[gid]`
/// records the base row each gid came from and every answer is mapped
/// back through it. SHAME(TALLFUNC)
async fn brute<T: DataPrimitive>(base: &Matrix<T>, query: &Matrix<T>, scratch: &str) -> Vec<Vec<(f32, u32)>>
where
    Point<T>: L2 + Data<Meta = PointMeta<T>>,
    <Point<T> as L2>::DistSq: Into<f64> + Copy + PartialOrd + Send + Sync,
{
    assert_eq!(base.dims, query.dims);
    let conf = BruteConfig { path: scratch.to_string(), dim: base.dims, mink: TOPK, maxk: TOPK, buflen: BUFLEN };
    let db = BruteDB::<TokioContext, T>::new(conf).await.unwrap();
    let meta = PointMeta::<T>::new(base.dims);
    let width = base.dims * size_of::<T>();
    let bbytes = unsafe { std::slice::from_raw_parts(base.data.as_ptr() as *const u8, size_of_val(&base.data[..])) };
    let qbytes = unsafe { std::slice::from_raw_parts(query.data.as_ptr() as *const u8, size_of_val(&query.data[..])) };
    for i in 0..query.rows {
        db.search(&pref(&meta, &qbytes[i * width..(i + 1) * width])).await;
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let rows: Vec<AtomicU32> = (0..base.rows).map(|_| AtomicU32::new(u32::MAX)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut work = pin!(async {
                    loop {
                        let row = next.fetch_add(1, Relaxed);
                        if row >= base.rows { break; }
                        let gid = db.insert(&pref(&meta, &bbytes[row * width..(row + 1) * width])).await;
                        let old = rows[gid as usize].swap(row as u32, Relaxed);
                        assert_eq!(old, u32::MAX, "gid {gid} handed out twice");
                    }
                });
                let mut cx = Context::from_waker(Waker::noop());
                while work.as_mut().poll(&mut cx).is_pending() {}
            });
        }
    });
    assert!(rows.iter().all(|row| row.load(Relaxed) != u32::MAX), "a base row was never inserted");
    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, Vec<(f32, u32)>)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads).map(|_| scope.spawn(|| {
            let mut work = pin!(async {
                let mut part = Vec::new();
                loop {
                    let i = next.fetch_add(1, Relaxed);
                    if i >= query.rows { break; }
                    let hits = db.search(&pref(&meta, &qbytes[i * width..(i + 1) * width])).await;
                    assert_eq!(hits.len(), TOPK, "query {i} answered with {} neighbors", hits.len());
                    let hits = hits.into_iter().map(|(dist, gid)| {
                        (dist.into() as f32, rows[gid as usize].load(Relaxed))
                    });
                    part.push((i, hits.collect()));
                }
                part
            });
            let mut cx = Context::from_waker(Waker::noop());
            loop {
                if let Poll::Ready(part) = work.as_mut().poll(&mut cx) { break part; }
            }
        })).collect();
        handles.into_iter().map(|handle| handle.join().unwrap()).collect()
    });
    let mut answer = vec![Vec::new(); query.rows];
    for (i, hits) in parts.into_iter().flatten() {
        answer[i] = hits;
    }
    answer
}

/// Write a ParlayANN ground truth file: `u32 nq, u32 k`, nq x k u32 ids, nq x k f32 distances
fn save(path: &str, answer: &[Vec<(f32, u32)>]) {
    let mut writer = BufWriter::new(File::create(path).unwrap());
    writer.write_all(&(answer.len() as u32).to_le_bytes()).unwrap();
    writer.write_all(&(TOPK as u32).to_le_bytes()).unwrap();
    for hits in answer {
        for &(_, id) in hits {
            writer.write_all(&id.to_le_bytes()).unwrap();
        }
    }
    for hits in answer {
        for &(dist, _) in hits {
            writer.write_all(&dist.to_le_bytes()).unwrap();
        }
    }
    writer.flush().unwrap();
}

/// `gtbrute <u8|i8|f32> <base> <query> <truth> <scratch>`
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (kind, base, query, truth, scratch) = (&args[1], &args[2], &args[3], &args[4], &args[5]);
    let mut cx = Context::from_waker(Waker::noop());
    let answer = match kind.as_str() {
        "u8" => {
            let (b, q) = (load::<u8>(base), load::<u8>(query));
            let mut work = pin!(brute(&b, &q, scratch));
            loop { if let Poll::Ready(done) = work.as_mut().poll(&mut cx) { break done; } }
        }
        "i8" => {
            let (b, q) = (load::<i8>(base), load::<i8>(query));
            let mut work = pin!(brute(&b, &q, scratch));
            loop { if let Poll::Ready(done) = work.as_mut().poll(&mut cx) { break done; } }
        }
        "f32" => {
            let (b, q) = (load::<f32>(base), load::<f32>(query));
            let mut work = pin!(brute(&b, &q, scratch));
            loop { if let Poll::Ready(done) = work.as_mut().poll(&mut cx) { break done; } }
        }
        other => panic!("unknown element type {other}"),
    };
    save(truth, &answer);
}
