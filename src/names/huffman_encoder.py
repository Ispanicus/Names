# source: https://www.geeksforgeeks.org/huffman-coding-greedy-algo-3/

import heapq
 
def huffman_encoder(token_freq: dict[str: int]) -> dict[str, int]:
    class node:
        def __init__(self, freq, symbol, left=None, right=None):
            self.freq = freq
            self.symbol = symbol
            self.left = left
            self.right = right
            self.huff = '' # tree direction (0/1)
            
        def __lt__(self, nxt):
            return self.freq < nxt.freq
            
    
    ## Build tree
    nodes = []
    for token, freq in token_freq.items():
        heapq.heappush(nodes, node(freq, token))
    
    while len(nodes) > 1:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
    
        # assign directional value to these nodes
        left.huff = 0
        right.huff = 1
    
        # combine the 2 smallest nodes to create
        # new node as their parent
        newNode = node(left.freq+right.freq, left.symbol+right.symbol, left, right)
        heapq.heappush(nodes, newNode)

    ## Create mapping from tree
    root = nodes[0]
    mapping = {}
    todo: list[node, str] = [(root, str(root.huff))]
    for nod, code in todo:
        code += str(nod.huff)
        if nod.left:
            todo.append((nod.left, code))
        if nod.right:
            todo.append((nod.right, code))
        if not nod.left and not nod.right:
            mapping[nod.symbol] = code

    return mapping
    
if __name__ == '__main__':
    token_freq = {'a': 5, 'b': 9, 'c': 12, 'd': 13, 'e': 16, 'f': 45}
    huffman_encoder(token_freq)
