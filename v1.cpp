#include <bits/stdc++.h>

using namespace std;

// https://codeforces.com/blog/entry/127488
int lcs(string X, string Y) {
  if (X.size() < Y.size()) swap(X, Y);
  int m = X.size(), n = Y.size();
  if (m == 0 || n == 0) return 0;
  int w = (m + 31) >> 5;
  std::uint32_t S[256][w];
  std::memset(S, 0, sizeof(std::uint32_t) * 256 * w);
  std::int32_t set = 1;
  for (int i = 0, j = 0; i < m; ++i) {
    S[X[i]][j] |= set;
    set <<= 1;
    if (!set) set++, j++;
  }
  std::uint32_t L[w];
  std::memset(L, 0, sizeof(std::uint32_t) * w);
  for (int i = 0; i < n; ++i) {
    std::uint32_t b1 = 1;
    std::uint32_t b2 = 0;
    for (int j = 0; j < w; ++j) {
      std::uint32_t U = L[j] | S[Y[i]][j];
      std::uint32_t c = L[j] >> 31;
      std::uint32_t V = U - (L[j] << 1 | b1 + b2);
      b1 = c;
      b2 = (V > U);
      L[j] = U & (~V);
    }
  }
  int res = 0;
  for (int i = 0; i < w; ++i)
    for (; L[i]; ++res) L[i] &= L[i] - 1;
  return res;
}

// int lcs(const string& a, const string& b) {
//   int n = a.size(), m = b.size();
//   vector<int> cur(m + 1);
//   // vector<vector<int>> dp(n + 1, vector<int>(m + 1));
//   for (int i = 1; i <= n; i++) {
//     vector<int> nxt(m + 1);
//     for (int j = 1; j <= m; j++) {
//       if (a[i - 1] == b[j - 1]) {
//         nxt[j] = cur[j - 1] + 1;
//       } else {
//         nxt[j] = max(nxt[j - 1], cur[j]);
//       }
//     }
//     cur.swap(nxt);
//   }
//   return cur[m];
// }

template <const int N>
vector<array<string, N>> read(string trainFile) {
  vector<array<string, N>> train;
  if (!fopen(trainFile.c_str(), "r")) return {};
  ifstream inp(trainFile.c_str());
  int n;
  inp >> n;
  train.resize(n);
  for (auto& it : train) {
    for (int i = 0; i < N; i++) inp >> it[i];
  }
  inp.close();
  return train;
}

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  freopen("answer.csv", "w", stdout);

  auto train = read<3>("cafa6/train.txt");
  auto test = read<3>("cafa6/test.txt");
  auto answer = read<2>("cafa6/answer.txt");

  cerr << "Read " << train.size() << " lines of train file\n";
  cerr << "Read " << test.size() << " lines of test file\n";
  cerr << "Read " << answer.size() << " lines of answer file\n";

  vector<array<string, 3>> newTrain;

  map<string, int> app;

  for (int i = 0; i < train.size(); i++) {
    if (app.count(train[i][2])) continue;
    newTrain.push_back(train[i]);
    app[train[i][2]] = 1;
  }

  train.swap(newTrain);
  cerr << "After filter: " << train.size() << endl;

  map<string, vector<int>> species_of;
  map<string, vector<string>> answer_of;
  for (int i = 0; i < train.size(); i++) {
    species_of[train[i][1]].emplace_back(i);
    app[train[i][2]] = i;
  }

  for (int i = 0; i < answer.size(); i++) {
    answer_of[answer[i][0]].emplace_back(answer[i][1]);
  }

  vector<int> all(train.size());
  iota(all.begin(), all.end(), 0);

  int cnt = 0;

  map<string, vector<string>> answer_for_gen;

  // Solve each test
  for (const auto& [id_gen, id_spec, gen] : test) {
    ++cnt;
    if ((cnt & 1023) == 0) {
      cerr << "Currently at test " << cnt << endl;
    }
    if (answer_for_gen.count(gen)) {
      for (auto it : answer_for_gen[gen]) {
        cout << id_gen << " " << it << " 1.000\n";
      }
      continue;
    }

    auto& cur_answer = answer_for_gen[gen];
    int cand = -1;
    if (app.count(gen))
      cand = app[gen];
    else {
      auto& ind = species_of[id_spec];
      if (ind.empty()) {
        ind = all;
      }
      tuple<int, int, int> best = {-1e9, -1e9, -1e9};
      for (const auto& cand : species_of[id_spec]) {
        auto value = lcs(gen, train[cand][2]);
        best = max(best, {value, -((int)train[cand][2].size()), cand});
      }
      // Use data of `cand`
      cand = get<2>(best);
    }
    assert(cand != -1e9);
    cur_answer = answer_of[train[cand][0]];
    for (auto it : cur_answer) {
      cout << id_gen << " " << it << " 1.000\n";
    }
  }
  return 0;
}