use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

/// A ParlayANN graph with every out-neighbor list packed back to back,
/// node `u` owning `target[offset[u]..offset[u + 1]]`
struct Graph {
    number: usize,
    maxdeg: usize,
    degree: Vec<u32>,
    offset: Vec<usize>,
    target: Vec<u32>,
}

/// Read a ParlayANN graph file\
/// `u32 n, u32 maxdeg, u32 degree[n]`, then each node's u32 neighbor ids in node order
fn load(path: &str) -> Graph {
    let mut reader = BufReader::with_capacity(1 << 24, File::open(path).unwrap());
    let mut words = |count: usize| {
        let mut data = vec![0u32; count];
        let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, count * 4) };
        reader.read_exact(bytes).unwrap();
        data
    };
    let head = words(2);
    let (number, maxdeg) = (head[0] as usize, head[1] as usize);
    let degree = words(number);
    let mut offset = Vec::with_capacity(number + 1);
    offset.push(0usize);
    for &deg in &degree {
        offset.push(offset.last().unwrap() + deg as usize);
    }
    let target = words(offset[number]);
    let mut rest = [0u8; 1];
    assert_eq!(reader.read(&mut rest).unwrap(), 0, "{path}: trailing bytes after edge lists");
    Graph { number, maxdeg, degree, offset, target }
}

/// Whether `v` lists `u` as an out-neighbor
fn mutual(graph: &Graph, u: u32, v: u32) -> bool {
    let v = v as usize;
    graph.target[graph.offset[v]..graph.offset[v + 1]].contains(&u)
}

/// Keep edge `u -> v` only when `v -> u` also exists\
/// node ids, node count and the max degree in the header stay unchanged
fn filter(graph: &Graph) -> Graph {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = graph.number.div_ceil(threads).max(1);
    let parts: Vec<(Vec<u32>, Vec<u32>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..graph.number).step_by(chunk).map(|lo| {
            scope.spawn(move || {
                let hi = (lo + chunk).min(graph.number);
                let mut degree = Vec::with_capacity(hi - lo);
                let mut target = Vec::new();
                for u in lo..hi {
                    let before = target.len();
                    for &v in &graph.target[graph.offset[u]..graph.offset[u + 1]] {
                        if mutual(graph, u as u32, v) {
                            target.push(v);
                        }
                    }
                    degree.push((target.len() - before) as u32);
                }
                (degree, target)
            })
        }).collect();
        handles.into_iter().map(|handle| handle.join().unwrap()).collect()
    });
    let mut degree = Vec::with_capacity(graph.number);
    let mut target = Vec::new();
    for (deg, tgt) in parts {
        degree.extend(deg);
        target.extend(tgt);
    }
    let mut offset = Vec::with_capacity(graph.number + 1);
    offset.push(0usize);
    for &deg in &degree {
        offset.push(offset.last().unwrap() + deg as usize);
    }
    Graph { number: graph.number, maxdeg: graph.maxdeg, degree, offset, target }
}

/// Write `graph` in the ParlayANN graph format [`load`] reads
fn save(path: &str, graph: &Graph) {
    let mut writer = BufWriter::with_capacity(1 << 24, File::create(path).unwrap());
    for part in [&[graph.number as u32, graph.maxdeg as u32][..], &graph.degree, &graph.target] {
        let bytes = unsafe { std::slice::from_raw_parts(part.as_ptr() as *const u8, part.len() * 4) };
        writer.write_all(bytes).unwrap();
    }
    writer.flush().unwrap();
}

/// Value of an unused slot in a flat graph row, as `0002/flatgraph.py` writes it
const PAD: u32 = u32::MAX;
/// Rows of a flat graph one filter step keeps and writes at once
const BLOCK: usize = 1 << 20;

/// [`filter`] over a flat graph, `width` u32 slots a row and [`PAD`] in unused ones, read whole
/// from `src` and written to `dst` in the same layout, [`BLOCK`] rows at a time over every core
fn flatfilter(width: usize, src: &str, dst: &str) {
    let bytes = std::fs::metadata(src).unwrap().len() as usize;
    assert_eq!(bytes % (width * 4), 0, "{src}: {bytes} bytes are no whole rows of {width} slots");
    let number = bytes / (width * 4);
    let mut rows = vec![0u32; number * width];
    let raw = unsafe { std::slice::from_raw_parts_mut(rows.as_mut_ptr() as *mut u8, bytes) };
    File::open(src).unwrap().read_exact(raw).unwrap();
    let rows = &rows;
    let row = move |v: usize| &rows[v * width..(v + 1) * width];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut writer = BufWriter::with_capacity(1 << 24, File::create(dst).unwrap());
    for lo in (0..number).step_by(BLOCK) {
        let hi = (lo + BLOCK).min(number);
        let mut out = vec![PAD; (hi - lo) * width];
        let part = (hi - lo).div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for (i, chunk) in out.chunks_mut(part * width).enumerate() {
                scope.spawn(move || {
                    for (r, slot) in chunk.chunks_mut(width).enumerate() {
                        let u = lo + i * part + r;
                        let kept = row(u).iter().filter(|&&v| v != PAD && row(v as usize).contains(&(u as u32)));
                        for (place, &v) in slot.iter_mut().zip(kept) {
                            *place = v;
                        }
                    }
                });
            }
        });
        let raw = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 4) };
        writer.write_all(raw).unwrap();
    }
    writer.flush().unwrap();
}

/// `bidir <ds> <src graph> <dst graph>`, or `bidir flat <width> <src flat> <dst flat>` for
/// [`flatfilter`]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args[1] == "flat" {
        flatfilter(args[2].parse().expect("width"), &args[3], &args[4]);
        return;
    }
    let (src, dst) = (&args[2], &args[3]);
    save(dst, &filter(&load(src)));
}
