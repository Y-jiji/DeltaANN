use std::str::FromStr;

/// A mix driver's command line, the same for every system: the dataset
/// label, its raw corpus `point`, flat `graph`, `query` file, `quant`izer
/// file, `index` path and event log `trace`, then key=value `knobs`
pub struct Command {
    pub label: String,
    pub point: String,
    pub graph: String,
    pub query: String,
    pub quant: String,
    pub index: String,
    pub trace: String,
    pub knobs: Vec<(String, String)>,
}

/// Parse `<dataset> <data> <graph> <queries> <quant> <index> <log>
/// [key=value ...]` from the process arguments\
/// **Behavior**\
/// panics on a missing file argument, a knob without `=`, or a key outside
/// `keys`
pub fn command(keys: &[&str]) -> Command {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() >= 7, "usage: <dataset> <data> <graph> <queries> <quant> <index> <log> [key=value ...]");
    let knobs = args[7..].iter().map(|arg| {
        let (key, value) = arg.split_once('=').unwrap_or_else(|| panic!("knob {arg} has no '='"));
        assert!(keys.contains(&key), "knob {key}, want one of {keys:?}");
        (key.to_string(), value.to_string())
    }).collect();
    let mut files = args.into_iter();
    let mut next = || files.next().unwrap_or_default();
    Command {
        label: next(),
        point: next(),
        graph: next(),
        query: next(),
        quant: next(),
        index: next(),
        trace: next(),
        knobs,
    }
}

/// The last value of knob `name` in `cmd` parsed, `fallback` when absent\
/// **Behavior**\
/// panics on a value that does not parse
pub fn knob<T: FromStr>(cmd: &Command, name: &str, fallback: T) -> T {
    match cmd.knobs.iter().rev().find(|(key, _)| key == name) {
        Some((_, text)) => text.parse().unwrap_or_else(|_| panic!("knob {name}={text} does not parse")),
        None => fallback,
    }
}
