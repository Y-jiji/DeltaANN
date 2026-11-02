use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write, copy};

/// Copy points `skip` to `skip + rows` of the headerless `src` into `dst`
/// behind a ParlayANN header `u32 rows, u32 dim`\
/// `size`: bytes per element
fn crop(src: &str, dst: &str, rows: usize, dim: usize, size: usize, skip: usize) {
    let bytes = (rows * dim * size) as u64;
    let start = (skip * dim * size) as u64;
    let have = std::fs::metadata(src).unwrap().len();
    assert!(have >= start + bytes, "{src}: {have} bytes, {skip} + {rows} x {dim} x {size} needs {}", start + bytes);
    assert_eq!(have % (dim * size) as u64, 0, "{src}: not a whole number of {dim} x {size} rows");
    let mut file = File::open(src).unwrap();
    file.seek(SeekFrom::Start(start)).unwrap();
    let reader = BufReader::with_capacity(1 << 24, file);
    let mut writer = BufWriter::with_capacity(1 << 24, File::create(dst).unwrap());
    writer.write_all(&(rows as u32).to_le_bytes()).unwrap();
    writer.write_all(&(dim as u32).to_le_bytes()).unwrap();
    let done = copy(&mut reader.take(bytes), &mut writer).unwrap();
    assert_eq!(done, bytes);
    writer.flush().unwrap();
}

/// `crop <src> <dst> <rows> <dim> <size> [skip]`
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num = |i: usize| -> usize { args[i].parse().unwrap() };
    let skip = if args.len() > 6 { num(6) } else { 0 };
    crop(&args[1], &args[2], num(3), num(4), num(5), skip);
}
