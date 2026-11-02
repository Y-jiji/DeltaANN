import re
import sys


def main(logs: list[str]) -> None:
    """Per log of `pqsweep`, the median-QPS search of every beam width, as
    `ds,graph,mode,book,beam,recall,qps`; `book` the codebook's letter, B for logs naming none."""
    print("ds,graph,mode,book,beam,recall,qps")
    for path in logs:
        text = open(path).read()
        ds, graph, mode = re.search(r"ds=(\S+) graph=(\S+) mode=(\S+)", text).groups()
        named = re.search(r"book=(?:\S*[./])?([AB])\.book", text)
        book = named.group(1) if named else "B"
        runs: dict[int, list[tuple[float, float]]] = {}
        for beam, recall, qps in re.findall(r"search: beam=(\d+) recall=(\S+) qps=(\S+)", text):
            runs.setdefault(int(beam), []).append((float(qps), float(recall)))
        for beam, found in runs.items():
            found.sort()
            qps, recall = found[len(found) // 2]
            print(f"{ds},{graph},{mode},{book},{beam},{recall},{qps}")


if __name__ == "__main__":
    main(sys.argv[1:])
