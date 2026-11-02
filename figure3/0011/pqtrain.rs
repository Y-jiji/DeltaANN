use std::env;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::str::FromStr;

use fnct::dataref::{DataMeta, DataPrimitive, SliceMeta};
use fnct::pt::{PointMeta, PqConf, ProductQBuilder};

/// Points the sample is spread over by default, env `BASE` overrides.
const BASE: usize = 100_000_000;
/// Dimensions per PQ band by default, env `BAND` overrides.
const BAND: usize = 4;
/// Code bits per PQ band by default, env `BITS` overrides.
const BITS: usize = 8;
/// Contiguous blocks the sample is read in, spread evenly over the base.
const BLOCKS: usize = 1000;
/// Points per sample block.
const BLOCKLEN: usize = 1000;
/// Lloyd iterations of training.
const ROUNDS: usize = 10;

/// Env `name` parsed, `default` when unset.
fn knob<T: FromStr>(name: &str, default: T) -> T {
    env::var(name).ok().map_or(default, |text| text.parse().ok().unwrap_or_else(|| panic!("parse {name}")))
}

/// Point element types a PQ codebook can be trained over.
trait Train: DataPrimitive + Sized {
    /// A PQ codebook of `band` dims and `bits` bits per band, trained by
    /// `ROUNDS` Lloyd iterations on `BLOCKS` blocks of `BLOCKLEN` points
    /// spread evenly over the first `base` points of the headerless `data`.
    fn train(data: &str, dim: usize, base: usize, band: usize, bits: usize) -> PqConf;
}

/// `Train` for each element type, which `ProductQBuilder` accepts.
macro_rules! train {
    ($($t:ty),*) => {$(
        impl Train for $t {
            fn train(data: &str, dim: usize, base: usize, band: usize, bits: usize) -> PqConf {
                let row = dim * std::mem::size_of::<$t>();
                let mut file = File::open(data).unwrap_or_else(|e| panic!("open {data}: {e}"));
                let mut bytes = vec![0u8; BLOCKS * BLOCKLEN * row];
                for (b, chunk) in bytes.chunks_exact_mut(BLOCKLEN * row).enumerate() {
                    file.seek(SeekFrom::Start((b * (base / BLOCKS) * row) as u64)).expect("seek sample");
                    file.read_exact(chunk).expect("read sample");
                }
                let meta = SliceMeta::new(BLOCKS * BLOCKLEN, PointMeta::<$t>::new(dim));
                let sample = meta.try_ref(&bytes);
                let mut builder = ProductQBuilder::<$t>::seeded(band, bits, 11, &sample);
                for r in 0..ROUNDS {
                    builder.iterate(&sample);
                    builder.finish();
                    eprintln!("pqtrain round={r}");
                }
                builder.intoconf()
            }
        }
    )*};
}

train!(u8, i8, f32);

/// Train on env `DATA` of env `DATASET` (sift 128 u8, spacev 100 i8, deep
/// 96 f32, msturing 100 f32) and write the codebook to env `OUTPUT` as raw
/// little-endian f32, env `BAND` and `BITS` setting the band shape.
fn main() {
    let dataset = env::var("DATASET").expect("set DATASET");
    let data = env::var("DATA").expect("set DATA");
    let output = env::var("OUTPUT").expect("set OUTPUT");
    let base = knob("BASE", BASE);
    let (band, bits) = (knob("BAND", BAND), knob("BITS", BITS));
    let conf = match dataset.as_str() {
        "sift" => u8::train(&data, 128, base, band, bits),
        "spacev" => i8::train(&data, 100, base, band, bits),
        "deep" => f32::train(&data, 96, base, band, bits),
        "msturing" => f32::train(&data, 100, base, band, bits),
        other => panic!("DATASET={other}, want sift|spacev|deep|msturing"),
    };
    let mut out = BufWriter::new(File::create(&output).unwrap_or_else(|e| panic!("create {output}: {e}")));
    for x in &conf.book { out.write_all(&x.to_le_bytes()).expect("write book"); }
    eprintln!("pqtrain dims={} band={} bits={} book={}", conf.dims, conf.band, conf.bits, conf.book.len());
}
