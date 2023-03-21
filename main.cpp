#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <igl/grad.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/min_quad_with_fixed.h>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <chrono>
#include "RichModel.h"

using namespace std;
using namespace igl;
using namespace Eigen;

MatrixXd V;
MatrixXi F;
SparseMatrix<double> G, L, M;

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
  int ei, fi; // find edge length through edge index
};

VectorXd calculate_scalar_value(const double t, const int src_point) {
  VectorXd u0 = VectorXd::Zero(V.rows());
  u0[0] = 1;
  auto start_time = chrono::system_clock::now();
  auto &&mid_tmp = MatrixXd(M-t*L);
  SparseMatrix<double> Q = M-t*L;
  SparseMatrix<double> _;
  min_quad_with_fixed_data<double> neumann;
  min_quad_with_fixed_precompute(Q, VectorXi(), _, true, neumann);
  VectorXd u;
  min_quad_with_fixed_solve(neumann, u0, VectorXd(), VectorXd(), u);
  // the line below will call 'std::bad_alloc' when the mesh is too large
  // VectorXd u = mid_tmp.householderQr().solve(u0);
  // VectorXd u = (mid_tmp.transpose()*mid_tmp).ldlt().solve(mid_tmp.transpose()*u0);
  auto end_time_1 = chrono::system_clock::now();
  chrono::duration<double> duration = (end_time_1-start_time);
  cout << "calculation time:" << duration.count() << endl;
  return u;
}
VectorXd normalized_gradient(VectorXd &u) {
  // let g be the gradient for each triangle faces
  VectorXd g = G * u;
  auto m = F.rows();
  for (int i = 0; i < F.rows(); ++i) {
    double norm = 0, max = -1;
    for (int j = 0; j < 3; ++j) {
      max = max > fabs(g[j*m+i])?max:fabs(g[j*m+i]);
    }
    for (int j = 0; j < 3; ++j) {
      norm += pow(g[j*m+i]/max, 2);
    }
    norm = sqrt(norm) * max;
    if (max == 0 || norm == 0) {
      for (int j = 0; j < 3; ++j) {
        g[j*m+i] = 0;
      }
    } else {
      for (int j = 0; j < 3; ++j) {
        g[j*m+i] /= norm;
      }
    }
  }
  return g;
}

inline double variance(VectorXd &vec) {
  const double mean = vec.mean();
  double var = 0;
  for (int i = 0; i < vec.rows(); ++i) {
    var += pow(vec[i]-mean, 2);
  }
  return var / vec.rows();
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cout << "<bin> <filename> <sigma_prefix>" << endl;
    exit(-1);
  }
  readOBJ(argv[1], V, F);
  // laplace cotan matrix
  cotmatrix(V, F, L);
  // mass mastrix for each triangle
  massmatrix(V, F, MASSMATRIX_TYPE_DEFAULT, M);
  // gradient operator
  grad(V, F, G);

  cout << "start calculation" << endl;

  const double t = pow(avg_edge_length(V, F), 2);
  VectorXd &&u = calculate_scalar_value(t, 0);

  cout << "scalar value calculation finished" << endl;

  VectorXd &&g = normalized_gradient(u);

  cout << "gradient normalization finished" << endl;

  Model3D::CRichModel model(argv[1]);
  model.LoadModel();
  // model.PrintInfo(cout);
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
      int actual_fi = model.Edge(model.GetEdgeIndexFromTwoVertices(vi, nv)).indexOfFrontFace;
      // no edge between nv and nnv
      if (actual_fi == -1) continue;
      auto &&vert_i = model.Vert(vi);
      auto &&vert_j = model.Vert(nv);
      auto &&vert_k = model.Vert(nnv);
      Vector3d v_i {vert_i.x, vert_i.y, vert_i.z};
      Vector3d v_j {vert_j.x, vert_j.y, vert_j.z};
      Vector3d v_k {vert_k.x, vert_k.y, vert_k.z};
      model.GetEdgeIndexFromTwoVertices(nv, nnv);
      perp_vec_base[vi].push_back({get_perp_vec_of_i(perp_point_of_jk(v_i, v_j, v_k), v_i), model.GetEdgeIndexFromTwoVertices(nv, nnv), actual_fi});
    }
  }

  // compute divergence
  VectorXd div_u = VectorXd::Zero(model.GetNumOfVerts());
  for (int vi = 0; vi < model.GetNumOfVerts(); ++vi) {
    for (auto &&val : perp_vec_base[vi]) {
      // find the normalized gradient vector of corresponding face
      Vector3d ux {g[val.fi], g[val.fi+F.rows()], g[val.fi, 2*F.rows()]};
      div_u[vi] += 0.5 * model.Edge(val.ei).length * ux.dot(val.perp_vec);
    }
  }

  cout << "calculation finished" << endl;

  double min = div_u[0], max = div_u[0];
  for (int i = 0; i < div_u.rows(); ++i) {
    min = min < div_u[i]?min:div_u[i];
    max = max > div_u[i]?max:div_u[i];
  }
  cout << "max:" << max << "\nmin:" << min << endl;
  double gap = abs(min), divider = max+abs(min);
  for (int i = 0; i < div_u.rows(); ++i) {
    div_u[i] += gap;
    div_u[i] /= divider;
  }

  // write div_u for data analysis
  fstream data_out;
  data_out.open("data.txt", ios::app | ios::out);
  for (int i = 0; i < V.rows(); ++i) {
    data_out << div_u[i] << "\n";
  }
  data_out.flush();
  data_out.close();

  // calculate the mean and variance of div_u
  const double mu = div_u.mean();
  const double var = variance(div_u);
  const double filter = mu-atoi(argv[2])*sqrt(var);
  set<int> base_collection;
  cout << "mean:" << mu << "\nvar:" << var << endl;
  // collect the vertices whose value is less than mu-3*sigma
  for (int i = 0; i < div_u.rows(); ++i) {
    if (div_u[i] < filter) {
      base_collection.insert(i);
      div_u[i] = 0;
    } else div_u[i] = 0.5;
  }

  // abandon sigle point condition
  for (auto vert : base_collection) {
    auto neigh = model.Neigh(vert);
    bool alone_in_neigh = true;
    for (auto neigh_edge : neigh) {
      bool is_left = (vert == model.Edge(neigh[0].first).indexOfLeftVert);
      int adjacent_vert;
      if (is_left) {
        adjacent_vert = model.Edge(neigh_edge.first).indexOfRightVert;
      } else {
        adjacent_vert = model.Edge(neigh_edge.first).indexOfLeftVert;
      }
      auto adjacent_in_collection = base_collection.find(adjacent_vert);
      if (adjacent_in_collection != base_collection.end()) {
        alone_in_neigh = false;
        break;
      }
    }
    if (alone_in_neigh) div_u[vert] = 0.5;
  }

  // write output model files
  fstream obj_out, mtl_out;
  obj_out.open("divergence_method.obj", ios::app | ios::out);
  mtl_out.open("divergence_method.mtl", ios::app | ios::out);
  obj_out << "mtllib divergence_method.mtl\nusemtl Default\ng default\n";
  for (int i = 0;i < V.rows();++i) {
    obj_out << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
  }
  for (int i = 0;i < V.rows();++i) {
    obj_out << "vt " << div_u(i) << " 0\n";
  }
  for (int i = 0;i < F.rows();++i) {
    obj_out <<"f "<<F(i, 0)+1<<"/"<<F(i, 0)+1<<" "<<F(i, 1)+1<<"/"<<F(i, 1)+1<<" "<<F(i, 2)+1<<"/"<<F(i, 2)+1<<"\n";
  }

  mtl_out << "newmtl Default\nKa 0 0 0\nKd 1 1 1\nKs 0.1 0.1 0.1\nmap_Kd colorbar.png\n";

  obj_out.flush();
  mtl_out.flush();
  obj_out.close();
  mtl_out.close();

  return 0;
}