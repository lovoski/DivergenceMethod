#pragma once
#include <vector>
#include <map>

using namespace std;

struct UnionFind
{
    vector<int> parents;
    vector<int> ranks;

    UnionFind() {}
    UnionFind(int n)
    {
        parents.clear();
        ranks.clear();
        parents.resize(n, 0);
        ranks.resize(n, 1);
        for (int i = 0; i < n; ++i) 
        {
            parents[i] = i;
            ranks[i] = 1;
        }
    }
    int FindAncestor(int id)
    {
        return (parents[id] == id) ? id : (parents[id] = FindAncestor(parents[id]));
    }

    void AddConnection(int x, int y)
    {
        int oldSize = parents.size();
        int newSize = max(x + 1, y + 1);
        if (newSize > oldSize)
        {
            parents.resize(newSize);
            ranks.resize(newSize);
            for (int i = oldSize; i < newSize; ++i)
            {
                parents[i] = i;
                ranks[i] = 1;
            }
        }

        x = FindAncestor(x);
        y = FindAncestor(y);
        if (x == y) return;
        if (ranks[x] > ranks[y]) 
            std::swap(x, y);
        parents[x] = y;
        if (ranks[x] == ranks[y])
            ranks[y] += 1;
    }

    void UpdateParents2Ancestors()
    {
        int n = parents.size();
        for (int i = 0; i < n; ++i)
        {  
            parents[i] = FindAncestor(i);
        }
    }

    map<int, vector<int>> GetAllClusters()
    {
        map<int, vector<int>> res;
        int n = parents.size();
        for (int i = 0; i < n; ++i)
        {
            int ancestor = FindAncestor(i);
            res[ancestor].push_back(i);
        }
        return res;
    }

    map<int, vector<int>> GetNontrivialClusters() 
    {
        map<int, vector<int>> res;
        int n = parents.size();
        for (int i = 0; i < n; ++i)
        {
            int ancestor = FindAncestor(i);
            if (ranks[ancestor] <= 1)
                continue;
            res[ancestor].push_back(i);
        }
        return res;
    }
    vector<int> GetLargestCluster()
    {
        set<pair<int, int>> ranks_from_small_to_large;
        int n = parents.size();
        for (int i = 0; i < n; ++i)
        {
            int ancestor = FindAncestor(i);
            ranks_from_small_to_large.insert(make_pair(ranks[ancestor], ancestor));          
        }
        vector<int> res;
        for (int i = 0; i < n; ++i)
        {
            int ancestor = FindAncestor(i);
            if (ancestor == ranks_from_small_to_large.rbegin()->second)
            {
                res.push_back(i);
            }
        }
        return res;
    }
};
