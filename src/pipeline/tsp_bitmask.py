'''
const int INF = 1e9;

int n; // number of cities
int dist[20][20]; // distance between cities
int memo[1 << 16][20]; // memoization table

int tsp(int mask, int pos) {
    if (mask == (1 << n) - 1) return dist[pos][0]; // all cities visited
    if (memo[mask][pos] != -1) return memo[mask][pos]; // already computed

    int ans = INF;
    for (int city = 0; city < n; city++) {
        if (mask & (1 << city)) continue; // city already visited
        int new_mask = mask | (1 << city);
        int new_ans = dist[pos][city] + tsp(new_mask, city);
        ans = min(ans, new_ans);
    }

    memo[mask][pos] = ans;
    return ans;
}

int main() {
    // read input
    cin >> n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> dist[i][j];
        }
    }

    // initialize memoization table
    memset(memo, -1, sizeof memo);

    // compute TSP
    int ans = tsp(1, 0); // start at city 0, with mask 1 (i.e. city 0 already visited)

    // output answer
    cout << ans << endl;

    return 0;
}
'''
INF = 1e9
n = 0

def tsp(mask:int, pos: int, dist, memo):
    if mask == (1 << n) - 1: return dist[pos][0]
    if memo[mask][pos] != -1: 
        return memo[mask][pos]
    
    ans = INF
    for city in range(n):
        if mask & (1 << city): continue # // city already visited
        new_mask = mask | (1 << city)
        a = tsp(new_mask, city, dist, memo)
        new_ans = dist[pos][city] + a
        if ans > new_ans:
            ans = new_ans

    

    memo[mask][pos] = ans
    return ans

import numpy as np
if __name__ == '__main__':
    n = int(input('No. of city: '))
    dist = np.zeros((n,n))

    for i in range(n):
        a = input().split(' ')
        for j in range(n):
            dist[i][j] = int(a[j])

    memo = np.zeros((2**n, n)) - 1

    ans = tsp(1, 0, dist, memo)

    print(f"Length: {ans}")

