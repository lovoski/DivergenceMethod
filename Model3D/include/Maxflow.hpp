// C++ implementation of Dinic's Algorithm
#include <vector>
#include <map>
#include <list>
using namespace std;

// Residual Graph
class Maxflow {

    // A structure to represent a edge between
// two vertex
    struct Edge {
        int v_target; // Vertex v (or "to" vertex)
        // of a directed edge u-v. "From"
        // vertex u can be obtained using
        // index in adjacent array.

        //int flow; // flow of data in edge

        int capacity; // capacity

        int indexOfReverseEdge; // To store index of reverse
        // edge in adjacency list so that
        // we can quickly find it.
    };

    vector<vector<Edge>> incidentEdges;

public:
    // add edge to the graph
    void AddEdge(int u, int v, int C)
    {
        //deal with insufficient space
        int necessaryspace = max(u, v) + 1;
        if (necessaryspace >= incidentEdges.size())
        {
            incidentEdges.resize(necessaryspace);
        }
        // Forward edge : 0 flow and C capacity
        Edge a{ v, C, (int)incidentEdges[v].size() };

        // Back edge : 0 flow and 0 capacity
        Edge b{ u, 0, (int)incidentEdges[u].size() };

        incidentEdges[u].push_back(a);
        incidentEdges[v].push_back(b); // reverse edge
    }
    int DinicMaxflow(int s, int t, map<pair<int, int>, int>& flow_plan) const;
    int DinicMaxflow(int s, int t) const;

protected:
    bool BFS(int s, int t, vector<int>&, const map<pair<int, int>, int>& flow_plan) const;
    int SendFlow(int s, int flow, int t, vector<int>&, const vector<int>& level, map<pair<int, int>, int>& flow_plan) const;
};

// Finds if more flow can be sent from s to t.
// Also assigns levels to nodes.
bool Maxflow::BFS(int s, int t, vector<int>& level, const map<pair<int, int>, int>& flow_plan) const
{
    int n = incidentEdges.size();
    level.clear();
    level.resize(n, -1);
    level[s] = 0; // Level of source vertex

    // Create a queue, enqueue source vertex
    // and mark source vertex as visited here
    // level[] array works as visited array also.
    list<int> q;
    q.push_back(s);

    vector<Edge>::const_iterator i;
    while (!q.empty()) {
        int u = q.front();
        q.pop_front();
        for (i = incidentEdges[u].begin(); i != incidentEdges[u].end(); i++) {
            const Edge& e = *i;

            if (level[e.v_target] < 0 && flow_plan.find(make_pair(u, i - incidentEdges[u].begin()))->second < e.capacity) {
                // Level of current vertex is,
                // level of parent + 1
                level[e.v_target] = level[u] + 1;

                q.push_back(e.v_target);
            }
        }
    }

    // IF we can not reach to the sink we
    // return false else true
    return level[t] < 0 ? false : true;
}

// A DFS based function to send flow after BFS has
// figured out that there is a possible flow and
// constructed levels. This function called multiple
// times for a single call of BFS.
// flow : Current flow send by parent function call
// start[] : To keep track of next edge to be explored.
//           start[i] stores  count of edges explored
//           from i.
//  u : Current vertex
//  t : Sink
int Maxflow::SendFlow(int u, int flow, int t, vector<int>& start, const vector<int>& level, map<pair<int, int>, int>& flow_plan) const
{
    // Sink reached
    if (u == t)
        return flow;

    // Traverse all adjacent edges one -by - one.
    for (; start[u] < incidentEdges[u].size(); start[u]++) {
        // Pick next edge from adjacency list of u
        const Edge& e = incidentEdges[u][start[u]];

        if (level[e.v_target] == level[u] + 1 && flow_plan.find(make_pair(u, start[u]))->second < e.capacity) {
            // find minimum flow from u to t
            int curr_flow = min(flow, e.capacity - flow_plan.find(make_pair(u, start[u]))->second);

            int temp_flow
                = SendFlow(e.v_target, curr_flow, t, start, level, flow_plan);

            // flow is greater than zero
            if (temp_flow > 0) {
                // add flow  to current edge
                flow_plan[make_pair(u, start[u])] += temp_flow;

                // subtract flow from reverse edge
                // of current edge
                flow_plan[make_pair(e.v_target, e.indexOfReverseEdge)] -= temp_flow;
                return temp_flow;
            }
        }
    }

    return 0;
}

// Returns maximum flow in graph
int Maxflow::DinicMaxflow(int s, int t, map<pair<int, int>, int>& flow_plan) const
{
    flow_plan.clear();
    for (int i = 0; i < incidentEdges.size(); ++i)
    {
        for (int j = 0; j < incidentEdges[i].size(); ++j)
        {
            flow_plan[make_pair(i, j)] = 0;
        }
    }
    // Corner case
    if (s == t)
        return 0;

    int total = 0; // Initialize result

    // Augment the flow while there is path
    // from source to sink
    int n = incidentEdges.size();
    vector<int> level(n, -1);
    while (BFS(s, t, level, flow_plan) == true) {
        // store how many edges are visited
        // from V { 0 to V }
        vector<int> start(n, 0);
        // while flow is not zero in graph from S to D
        while (int flow = SendFlow(s, INT_MAX, t, start, level, flow_plan))

            // Add path flow to overall flow
            total += flow;
    }

    // return maximum flow
    return total;
}

int Maxflow::DinicMaxflow(int s, int t) const
{
    map<pair<int, int>, int> flow_plan;
    return DinicMaxflow(s, t, flow_plan);
}
