//
// Created by guanghere on 25-4-22.
//
#include <climits>
#include <complex>
#include <iostream>
#include <queue>
#include <vector>

class MinCostMaxFlow {
    const int INF = INT_MAX;

   public:
    explicit MinCostMaxFlow(int n)
        : n(n), adj(n + 1), dis(n + 1), vis(n + 1), cur(n + 1), ret(0) {}

    void add_edge(int u, int v, double cap, double cost) {
        adj[u].emplace_back(
            Edge{v, cap, cost, static_cast<int>(adj[v].size()), true});
        adj[v].emplace_back(
            Edge{u, 0, -cost, static_cast<int>(adj[u].size()) - 1, false});
    }

    double min_cost_max_flow(int s, int t) {
        double max_flow = 0;
        while (spfa(s, t)) {
            std::fill(vis.begin(), vis.end(), false);
            std::fill(cur.begin(), cur.end(),
                      0);  // Reset current arc pointers after SPFA
            double flow;
            while ((flow = dfs(s, t, INF))) {
                max_flow += flow;
            }
        }
        return max_flow;
    }

    double get_cost() const { return ret; }

   private:
    struct Edge {
        int to;
        double cap;
        double cost;
        int rev;
        bool is_positive;
    };

    int n;
    double ret;
    std::vector<std::vector<Edge>> adj;
    std::vector<double> dis;
    std::vector<int> cur;  // cur array for current arc optimization
    std::vector<bool> vis;

    bool spfa(int s, int t) {
        std::fill(dis.begin(), dis.end(), INF);
        dis[s] = 0;
        std::queue<int> q;
        q.push(s);
        vis[s] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            vis[u] = false;

            for (const auto &e : adj[u]) {
                if (e.cap > 0 && dis[e.to] > dis[u] + e.cost) {
                    dis[e.to] = dis[u] + e.cost;
                    if (!vis[e.to]) {
                        q.push(e.to);
                        vis[e.to] = true;
                    }
                }
            }
        }
        return dis[t] != INF;
    }

    int dfs(int u, int t, double flow) {
        if (u == t) return flow;
        vis[u] = true;
        double total_flow = 0;

        // Iterate from cur[u], inspired by Dinic's ptr[u]
        for (int &i = cur[u]; i < adj[u].size(); ++i) {
            auto &e = adj[u][i];
            if (!vis[e.to] && e.cap > 0 && dis[e.to] == dis[u] + e.cost) {
                double pushed = dfs(e.to, t, std::min(flow - total_flow, e.cap));
                if (pushed > 0) {
                    e.cap -= pushed;
                    adj[e.to][e.rev].cap += pushed;
                    ret += pushed * e.cost;
                    total_flow += pushed;
                    if (total_flow == flow) break;
                }
            }
        }
        vis[u] = false;
        return total_flow;
    }
};

double dist(double x1, double y1, double x2, double y2) {
    return std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
}

const int inf = std::numeric_limits<int>::max();

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    int x1 = 5, y1 = 1;  // a
    int x2 = 2, y2 = 7;  // b
    std::vector<std::vector<double>> sites = {{1.25, 1.25}, {8.75, 0.75},
                                              {0.5, 4.75},  {5.75, 5},
                                              {3, 6.5},     {7.25, 7.25}};
    std::vector<int> demands = {3, 5, 4, 7, 6, 11};

    MinCostMaxFlow mcmf(2 + 2 + 6);
    int s = 7, t = 8, a = 9, b = 10;
    mcmf.add_edge(s, a, 20, 0);
    mcmf.add_edge(s, b, 20, 0);
    for (int i = 1; i <= 6; i++) {
        mcmf.add_edge(a, i, demands[i - 1],
                      dist(x1, y1, sites[i - 1][0], sites[i - 1][1]));
        mcmf.add_edge(b, i, demands[i - 1],
                      dist(x2, y2, sites[i - 1][0], sites[i - 1][1]));
        mcmf.add_edge(i, t, demands[i - 1], 0);
    }
    std::cout << mcmf.min_cost_max_flow(s, t) << '\n';
    std::cout << mcmf.get_cost();
}