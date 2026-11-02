use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write, copy};

/// Copy the first `rows` points of the headerless `src` into `dst`
/// behind a ParlayANN header `u32 rows, u32 dim`\
/// `size`: bytes per element
fn crop(src: &str, dst: &str, rows: usize, dim: usize, size: usize) {
    let bytes = (rows * dim * size) as u64;
    let have = std::fs::metadata(src).unwrap().len();
    assert!(have >= bytes, "{src}: {have} bytes, {rows} x {dim} x {size} needs {bytes}");
    assert_eq!(have % (dim * size) as u64, 0, "{src}: not a whole number of {dim} x {size} rows");
    let reader = BufReader::with_capacity(1 << 24, File::open(src).unwrap());
    let mut writer = BufWriter::with_capacity(1 << 24, File::create(dst).unwrap());
    writer.write_all(&(rows as u32).to_le_bytes()).unwrap();
    writer.write_all(&(dim as u32).to_le_bytes()).unwrap();
    let done = copy(&mut reader.take(bytes), &mut writer).unwrap();
    assert_eq!(done, bytes);
    writer.flush().unwrap();
}

/// `crop <src> <dst> <rows> <dim> <size>`
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num = |i: usize| -> usize { args[i].parse().unwrap() };
    crop(&args[1], &args[2], num(3), num(4), num(5));
}
