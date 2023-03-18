#include <chrono>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "RichModel.h"
using namespace std;
using namespace Eigen;

// structure that stores the vector
struct edge_normal_vec {
  int fi;
  Vector3d n;
};

// only consider a closed mesh
int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "<bin> <model_filename>" << endl;
    exit(-1);
  }
  Model3D::CRichModel model(argv[1]);
  // calculate normal vector for each edge
  // each edge normal is in the surrounding area of a vertex
  // query the surrounding area for each vertex
  // for (int vi = 0; vi < model.GetNumOfVerts(); ++vi) {
  //   auto &&ele = model.Neigh(vi);
  // }
  for (auto &&ele : model.Neigh(0)) {
    cout << ele.first << endl;
  }

  return 0;
}