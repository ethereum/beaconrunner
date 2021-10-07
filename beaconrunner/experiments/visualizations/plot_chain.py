import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz
from networkx.drawing.nx_agraph import pygraphviz_layout

def get_node_index(tree, block_root):
    return [x for x,y in tree.nodes(data=True) if y['block_root']==block_root][0]

def get_parent_node_index(tree, block_headers, block_index):
    parent_block_root = block_headers[block_index].parent_root
    parent_node_index = get_node_index(tree, parent_block_root)
    return parent_node_index

def tag_chain(network, block_headers, tree):
    latest_block_root = network.validators[0].get_head()
    last_block_index = [x for x,y in tree.nodes(data=True) if y['block_root']==latest_block_root][0]

    canonical_chain = [last_block_index]
    query_next = last_block_index
    while True:
        parent = get_parent_node_index(tree, block_headers, query_next)
        canonical_chain.insert(0, parent)
        if parent == 0: #genesis
            break
        query_next = parent

    orphaned_chain = list(set(list(range(canonical_chain[-1]+1))) - set(canonical_chain))

    return canonical_chain, orphaned_chain

def get_chain_tree(network):
    blocks_dict = network.validators[0].store.blocks # access blocks from validator's Store

    block_headers = list(blocks_dict.values()) # extract block headers
    block_roots = list(blocks_dict) # extract block roots

    tree = nx.DiGraph()

    # Add all blocks as nodes to directed graph
    for index, block_header in enumerate(block_headers):
        tree.add_nodes_from([
        (index, {"slot": block_header.slot, "parent_root": block_header.parent_root, "block_root": block_roots[index]})
        ])

    # Add edges between blocks respectively
    for i in range(len(tree)):
        for j in range(len(tree)):
            if tree.nodes()[i]["parent_root"] == tree.nodes()[j]["block_root"]:
                tree.add_edge(i, j)
    
    canonical_chain_ids, orphaned_chain_ids = tag_chain(network, block_headers, tree)
    
    return block_headers, tree, canonical_chain_ids, orphaned_chain_ids

def plot_chain_tree(network):
    block_headers, tree, canonical_chain_ids, orphaned_chain_ids = get_chain_tree(network)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), sharex=True, sharey=True)

    pos = pygraphviz_layout(tree, prog='dot', args='-Grankdir="RL"')
    for i in pos:
        pos[i] = (block_headers[i].slot + 200, pos[i][1])

    plt.title("Tree of blocks")

    nx.draw_networkx_nodes(tree,pos=pos, nodelist=canonical_chain_ids, node_color='green', label="Canonical blocks", node_shape='s', node_size=500)
    nx.draw_networkx_nodes(tree,pos=pos, nodelist=orphaned_chain_ids, node_color='orange', label="Orphaned blocks", node_shape='s', node_size=500)
    nx.draw_networkx_nodes(tree,pos=pos, nodelist=[0], node_color='blue', label="Genesis block", node_shape='s', node_size=500)    

    nx.draw_networkx_edges(tree, pos, arrows=True)
    nx.draw_networkx_labels(tree, pos, font_size=10, font_color="white")

    ax.set_frame_on(False)
    plt.legend(loc="lower right", labelspacing=2, fontsize=9, frameon=False, borderpad=0.1)

    return ax