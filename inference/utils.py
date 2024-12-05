import numpy as np
import cv2
import pdb


def visualize(im, list_segments, sorted_indexes, save_path, scale=1.0):
    sorted_segments = [list_segments[i] for i in sorted_indexes]
    im = im.copy()
    im_h, im_w = im.shape[:2]
    if scale:
        im = cv2.resize(im, (int(im_w * scale), int(im_h * scale)))
    for i, segment_info in enumerate(sorted_segments):
        order = i
        bb = segment_info['p4_bb']
        if scale:
            bb = list(map(lambda x: int(x * scale), bb))
        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)
        cv2.putText(im, str(order), (bb[0], bb[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(save_path, im)



def greedy_postprocess(prob_matrix):
    """
        probs: np array shape (n,n)
    """
    n = prob_matrix.shape[0] - 2
    sorted_indexes = [np.argmax(prob_matrix[0])]
    while len(sorted_indexes) < n:
        scores = prob_matrix[sorted_indexes[-1]]
        cands_list = np.argsort(scores)[::-1]
        for index in cands_list:
            if index not in sorted_indexes:
                sorted_indexes.append(index)
                break
        # print('sorted_indexes: ', sorted_indexes)
        # pdb.set_trace()
        
    sorted_indexes = [index-1 for index in sorted_indexes]
    sorted_indexes = [index for index in sorted_indexes if 0 <= index < n]
    for i in range(n):
        if i not in sorted_indexes:
            sorted_indexes.append(i)
    # pdb.set_trace()
    assert set(sorted_indexes) == set(list(range(n)))
    return sorted_indexes
    
    

def greedy_best_path(prob_matrix):
    """
    Greedy algorithm to find a path through the graph maximizing the probability product.
    
    Args:
    prob_matrix (np.ndarray): The n x n probability matrix.
    
    Returns:
    tuple: (best_path, max_probability_product)
    """
    n = prob_matrix.shape[0]
    visited = [False] * n
    path = [0]  # Start at <cls>
    visited[0], visited[n-1] = True, True
    product = 1.0
    list_probs = []

    # Greedily select the next highest probability edge
    for _ in range(1, n - 1):  # Exclude the start and end nodes
        last = path[-1]
        next_node = -1
        max_prob = 0

        for j in range(n-1):
            if not visited[j] and prob_matrix[last, j] > max_prob:
                next_node = j
                max_prob = prob_matrix[last, j]
        
        if next_node == -1:
            raise ValueError("Graph is disconnected.")
        
        path.append(next_node)
        visited[next_node] = True
        product *= max_prob
        list_probs.append(max_prob)

    # Finish at <sep> (last node)
    path.append(n - 1)
    product *= prob_matrix[path[-2], n - 1]
    list_probs.append(prob_matrix[path[-2], n - 1])
    
    # pdb.set_trace()
    path = path[1:-1]
    path = [el-1 for el in path]
    return path, list_probs




def find_best_path_with_expansion(prob_matrix, threshold=0.9, top_k=3):
    n = len(prob_matrix)
    
    def dfs(path, visited, prob):
        current_node = path[-1]
        
        # If we reached the end node <sep>, check if the path is valid
        print(f'path: {path}')
        if len(path) == n-1:
            return path, prob
        
        # Get probabilities for the current node
        probs = prob_matrix[current_node]
        
        # Filter unvisited nodes
        candidates = [(i, p) for i, p in enumerate(probs) if i not in visited]
        if not candidates:
            return None, 0  # Dead-end
        
        # Sort candidates by probability
        candidates.sort(key=lambda x: -x[1])
        
        # Branch based on the highest probability
        if candidates[0][1] > threshold:
            # Greedy step: pick the node with the highest probability
            next_node = candidates[0][0]
            return dfs(path + [next_node], visited | {next_node}, prob * candidates[0][1])
        else:
            # Explore top-k nodes if highest probability is below threshold
            best_path, best_prob = None, 0
            for i in range(min(top_k, len(candidates))):
                next_node, next_prob = candidates[i]
                new_path, new_prob = dfs(path + [next_node], visited | {next_node}, prob * next_prob)
                if new_prob > best_prob:
                    best_path, best_prob = new_path, new_prob
            return best_path, best_prob

    # Start the search from <cls> node (node 0)
    best_path, best_prob = dfs([0], {0, n-1}, 1)
    best_path = [el - 1 for el in best_path[1:]]
    assert len(best_path) == n-2
    return best_path, best_prob





if __name__ == '__main__':
    import json
    import cv2

    ip = 'data/CompHRDoc_reduce/val/1507.02346_0.png'
    jp = 'data/CompHRDoc_reduce/val/1507.02346_0.json'
    im = cv2.imread(ip)
    with open(jp) as f:
        list_segments = json.load(f)
    
    visualize(im, list_segments, list(range(len(list_segments))), 'test.jpg', 2.0)