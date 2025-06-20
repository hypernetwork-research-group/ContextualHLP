import os
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
import hypernetx as hnx
from dotenv import load_dotenv
from openai import OpenAI


def load_patent_data(csv_path: Path, max_rows: int = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.head(max_rows) if max_rows else df


def extract_inventor_mappings(
    df: pd.DataFrame, inventor_cols: list[str]
) -> dict[str, list[str]]:
    mapping = {}
    for _, row in df.iterrows():
        patent_id = str(row['patent_number'])
        inventors = [row[col] for col in inventor_cols if pd.notna(row[col])]
        if len(inventors) >= 2:
            mapping[patent_id] = inventors
    return mapping


def invert_mapping(mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    inv_map = defaultdict(list)
    for patent, inventors in mapping.items():
        for inv in inventors:
            inv_map[inv].append(patent)
    return dict(inv_map)


def filter_single_patent_inventors(inv_map: dict[str, list[str]]) -> list[str]:
    return [inv for inv, patents in inv_map.items() if len(patents) > 1]


def build_hypergraph(patent_map: dict[str, list[str]]) -> hnx.Hypergraph:
    edges = {patent: set(inventors) for patent, inventors in patent_map.items()}
    return hnx.Hypergraph(edges)


def largest_connected_component_subgraph(H: hnx.Hypergraph) -> hnx.Hypergraph:
    comps = list(H.connected_components())
    if not comps:
        raise ValueError("No connected components found in hypergraph.")
    largest = max(comps, key=len)
    return H.restrict_to_nodes(largest)


def relabel_entities(subH: hnx.Hypergraph) -> tuple[dict[str, int], dict[str, int], dict[int, list[int]]]:
    nodes = list(subH.nodes())
    node_map = {name: idx for idx, name in enumerate(nodes)}

    edges = list(subH.edges())
    edge_map = {pid: idx for idx, pid in enumerate(edges)}

    hypergraph = {
        edge_map[pid]: [node_map[n] for n in subH.edges(pid)]
        for pid in edges
    }
    return node_map, edge_map, hypergraph


def export_hypergraph(
    hypergraph: dict[int, list[int]],
    node_map: dict[str, int],
    edge_map: dict[str, int],
    output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_path = output_dir / 'hypergraph.txt'
    with edges_path.open('w') as f:
        for neigh_list in hypergraph.values():
            f.write(" ".join(map(str, neigh_list)) + "\n")

    with (output_dir / 'nodes.pkl').open('wb') as f:
        pickle.dump(node_map, f)
    with (output_dir / 'edges.pkl').open('wb') as f:
        pickle.dump(edge_map, f)


def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    csv_file = Path('src/data2/2023.csv')
    max_rows = 50_000
    inventor_cols = [f'inventor_name{i}' for i in range(1, 10)]
    output_dir = Path('src/data2/')

    df = load_patent_data(csv_file, max_rows)
    patent_to_inventors = extract_inventor_mappings(df, inventor_cols)
    inventor_to_patents = invert_mapping(patent_to_inventors)

    H = build_hypergraph(patent_to_inventors)
    subH = largest_connected_component_subgraph(H)

    node_map, edge_map, hypergraph = relabel_entities(subH)
    export_hypergraph(hypergraph, node_map, edge_map, output_dir)

    print(f"Processed {len(df)} patents; exported hypergraph with \
          {len(node_map)} nodes and {len(edge_map)} edges.")


if __name__ == '__main__':
    main()

