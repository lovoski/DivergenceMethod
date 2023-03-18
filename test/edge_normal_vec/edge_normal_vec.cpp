#include <chrono>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "RichModel.h"
using namespace std;
using namespace Eigen;

// get perpendicular point of edge ik in triangle ijk
inline Vector3d perp_point_of_jk(Vector3d i, Vector3d j, Vector3d k) {
  const double t = ((k.x()-j.x())*(k.x()-i.x())+(k.y()-j.y())*(k.y()-i.y())+(k.z()-j.z())*(k.z()-i.z()))
    /((j.x()-k.x())*(j.x()-k.x())+(j.y()-k.y())*(j.y()-k.y())+(j.z()-k.z())*(j.z()-k.z()));
  return t*j+(1-t)*k;
}
// the perp_vec should go through vertex i and perp
// this should also be a normalized vector
inline Vector3d get_perp_vec_of_i(Vector3d &perp, Vector3d &i) {
  Vector3d perp_vec = perp-i;
  double max = fabs(perp_vec[0]);
  max = fabs(perp_vec[1])>max?fabs(perp_vec[1]):max;
  max = fabs(perp_vec[2])>max?fabs(perp_vec[2]):max;
  double mu = 0;
  for (int i = 0; i < 3; ++i) mu += pow(perp_vec[i]/max, 2);
  mu = sqrt(mu)*max;
  for (int i = 0; i < 3; ++i) perp_vec[i] /= mu;
  return perp_vec;
}

struct perp_vec_data {
  Vector3d perp_vec;
  int ei; // find edge length through edge index
};

// only consider a closed mesh
int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "<bin> <model_filename>" << endl;
    exit(-1);
  }
  Model3D::CRichModel model(argv[1]);
  model.LoadModel();
  model.PrintInfo(cout);
  // calculate perpendicular point for each edge in every triangle
  vector<vector<perp_vec_data>> perp_vec_base;
  perp_vec_base.resize(model.GetNumOfVerts());
  for (int vi = 0; vi < model.GetNumOfVerts(); ++vi) {
    auto &&ne = model.Neigh(vi);
    // follow the default order
    bool is_left_vert = (model.Edge(ne[0].first).indexOfLeftVert == vi);
    for (size_t i = 0; i < ne.size(); ++i) {
      int nv, nnv;
      if (is_left_vert) {
        nv = model.Edge(ne[i].first).indexOfRightVert;
        nnv = model.Edge(ne[(i+1)%ne.size()].first).indexOfRightVert;
      } else {
        nv = model.Edge(ne[i].first).indexOfLeftVert;
        nnv = model.Edge(ne[(i+1)%ne.size()].first).indexOfLeftVert;
      }
      auto &&vert_i = model.Vert(vi);
      auto &&vert_j = model.Vert(nv);
      auto &&vert_k = model.Vert(nnv);
      Vector3d v_i {vert_i.x, vert_i.y, vert_i.z};
      Vector3d v_j {vert_j.x, vert_j.y, vert_j.z};
      Vector3d v_k {vert_k.x, vert_k.y, vert_k.z};
      model.GetEdgeIndexFromTwoVertices(nv, nnv);
      perp_vec_base[vi].push_back({get_perp_vec_of_i(perp_point_of_jk(v_i, v_j, v_k), v_i), model.GetEdgeIndexFromTwoVertices(nv, nnv)});
    }
  }
  // fstream out;
  // remove("line.txt");
  // out.open("line.txt", ios::out | ios::app);
  // auto ne = model.Neigh(0);
  // for (auto &&e : ne) {
  //   auto li = model.Edge(e.first).indexOfLeftVert;
  //   auto ri = model.Edge(e.first).indexOfRightVert;
  //   cout << "li:" << li << " ri:" << ri << "\n";
  //   out << "l " << 0 << " " << ri << "\n";
  // }
  // out.flush();
  // out.close();
  return 0;
}