import numpy as np
import pandas as pd
import networkx as nx


def build_graph_for_window(meta_df: pd.DataFrame):
    G = nx.DiGraph()
    for _, row in meta_df.iterrows():
        src = f"src:{row.get('src_ip','unknown_src')}"
        dst = f"dst:{row.get('dst_ip','unknown_dst')}"
        service = f"svc:{row.get('service', row.get('dst_port','unknown_svc'))}"
        G.add_node(src, ntype='src')
        G.add_node(dst, ntype='dst')
        G.add_node(service, ntype='service')
        weight = float(row.get('bytes', row.get('packets', 1.0)) or 1.0)
        if G.has_edge(src, service):
            G[src][service]['weight'] += weight
        else:
            G.add_edge(src, service, weight=weight)
        if G.has_edge(service, dst):
            G[service][dst]['weight'] += weight
        else:
            G.add_edge(service, dst, weight=weight)
    return G


def extract_graph_descriptors(G):
    n = G.number_of_nodes(); e = G.number_of_edges()
    if n == 0:
        return np.zeros(8, dtype='float32')
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    weights = np.array([d.get('weight',1.0) for _,_,d in G.edges(data=True)], dtype=float) if e else np.array([0.0])
    und = G.to_undirected()
    density = nx.density(G) if n > 1 else 0.0
    clustering = nx.average_clustering(und) if n > 1 else 0.0
    desc = np.array([
        n, e, degrees.mean(), degrees.var(), density, clustering, weights.mean(), weights.var()
    ], dtype='float32')
    return np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)


def descriptors_from_windows(df, starts, T=30):
    desc = []
    for st in starts:
        G = build_graph_for_window(df.iloc[int(st):int(st)+T])
        desc.append(extract_graph_descriptors(G))
    return np.vstack(desc).astype('float32')
