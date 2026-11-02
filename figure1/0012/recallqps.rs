/// Dataset, graph and every `(beam, recall, qps)` search a `sweep` log holds\
/// the first `ds=<ds> graph=<graph>` line names the log, each `search:` line is one run
fn parse(path: &str) -> (String, String, Vec<(u32, f64, f64)>) {
    let text = std::fs::read_to_string(path).unwrap();
    let field = |line: &str, key: &str| -> String {
        line.split([' ', ',']).find_map(|word| word.strip_prefix(key)).unwrap().to_string()
    };
    let head = text.lines().find(|line| line.starts_with("ds=")).unwrap();
    let (ds, graph) = (field(head, "ds="), field(head, "graph="));
    let runs = text.lines().filter(|line| line.starts_with("search:")).map(|line| {
        let beam = field(line, "Q=").parse().unwrap();
        let recall = field(line, "recall=").parse().unwrap();
        let qps = field(line, "QPS=").parse().unwrap();
        (beam, recall, qps)
    }).collect();
    (ds, graph, runs)
}

/// Per beam width, the run with the median QPS, in sweep order
fn median(runs: &[(u32, f64, f64)]) -> Vec<(u32, f64, f64)> {
    let mut beams: Vec<u32> = runs.iter().map(|run| run.0).collect();
    beams.dedup();
    beams.iter().map(|&beam| {
        let mut same: Vec<(u32, f64, f64)> = runs.iter().copied().filter(|run| run.0 == beam).collect();
        same.sort_by(|a, b| a.2.total_cmp(&b.2));
        same[same.len() / 2]
    }).collect()
}

/// `recallqps <log>...`, prints `ds,graph,beam,recall,qps`
fn main() {
    println!("ds,graph,beam,recall,qps");
    for path in std::env::args().skip(1) {
        let (ds, graph, runs) = parse(&path);
        for (beam, recall, qps) in median(&runs) {
            println!("{ds},{graph},{beam},{recall},{qps}");
        }
    }
}
