#include <iostream>
#include <cstdio>
#include <fstream>
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
void calculate_divegence() {}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cout << "<bin> <filename> <t>" << endl;
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

  VectorXd &&u = calculate_scalar_value(atof(argv[2]), 0);

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

  // fstream out;
  // remove("out.txt");
  // out.open("out.txt", ios::app | ios::out);
  // for (int i = 0; i < div_u.rows(); ++i) {
  //   out << div_u[i] << "\n";
  // }
  // out.flush();
  // out.close();

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

  // visualize the data
  // opengl::glfw::Viewer viewer;
  // viewer.data().set_mesh(V, F);

  // // barycenter is a face_num * 3 matrix
  // MatrixXd BC;
  // barycenter(V, F, BC);
  // MatrixXd T = MatrixXd::Zero(BC.rows(), BC.cols());
  // for (int i = 0; i < F.rows(); ++i) {
  //   T(i, 0) = g[i+0*F.rows()];
  //   T(i, 1) = g[i+1*F.rows()];
  //   T(i, 2) = g[i+2*F.rows()];
  // }
  // const double length = avg_edge_length(V, F)*0.5;
  // // write each gradeint vector to a file
  // fstream out;
  // out.open("out.txt", ios::out | ios::app);
  // for (int i = 0; i < F.rows(); ++i) {
  //   // write format: start_x,start_y,start_z,end_x,end_y,end_z
  //   out << BC(i, 0) << "," << BC(i, 1) << "," << BC(i, 2) << "," <<
  //      BC(i, 0)+T(i, 0) << "," << BC(i, 1)+T(i, 1) << "," << BC(i, 2)+T(i, 2) << "\n";
  // }
  // out.flush();
  // out.close();
  return 0;
}