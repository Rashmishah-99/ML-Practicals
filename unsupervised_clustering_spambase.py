import csv
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


def load_spambase(path):
    with open(path, encoding='utf-8', errors='replace') as fp:
        reader = csv.reader(fp)
        header = next(reader)
        data = []
        labels = []
        for row in reader:
            if not row:
                continue
            data.append([float(x) for x in row[:-1]])
            labels.append(int(float(row[-1])))
    return np.array(data), np.array(labels), header


def best_cluster_accuracy(true_labels, cluster_labels):
    labels = np.unique(true_labels)
    clusters = np.unique(cluster_labels)
    cost_matrix = np.zeros((len(labels), len(clusters)), dtype=int)
    for i, label in enumerate(labels):
        for j, cluster in enumerate(clusters):
            cost_matrix[i, j] = np.sum((true_labels == label) & (cluster_labels == cluster))
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    matched = {clusters[col]: labels[row] for row, col in zip(row_ind, col_ind)}
    mapped = np.array([matched[c] for c in cluster_labels])
    return accuracy_score(true_labels, mapped)


def evaluate_clustering(name, model, X, y_true):
    labels_pred = model.fit_predict(X)
    acc = best_cluster_accuracy(y_true, labels_pred)
    ari = adjusted_rand_score(y_true, labels_pred)
    nmi = normalized_mutual_info_score(y_true, labels_pred)
    print(f"{name} results:")
    print(f"  Accuracy (best label mapping): {acc:.4f}")
    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"  Normalized Mutual Info: {nmi:.4f}\n")
    return {'name': name, 'accuracy': acc, 'ari': ari, 'nmi': nmi}


def main():
    path = 'spambase.csv'
    X, y, header = load_spambase(path)
    print(f"Loaded {X.shape[0]} rows and {X.shape[1]} features from {path}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    results.append(evaluate_clustering('KMeans (k=2)', KMeans(n_clusters=2, random_state=42, n_init=10), X_scaled, y))
    results.append(evaluate_clustering('AgglomerativeClustering (2 clusters)', AgglomerativeClustering(n_clusters=2), X_scaled, y))

    best = max(results, key=lambda r: r['ari'])
    print('Best clustering by ARI:')
    print(f"  {best['name']} with ARI={best['ari']:.4f}, Accuracy={best['accuracy']:.4f}, NMI={best['nmi']:.4f}")

if __name__ == '__main__':
    main()
