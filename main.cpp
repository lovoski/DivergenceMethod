#include <iostream>
#include <fstream>
#include <string>
#include <igl/grad.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
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

VectorXd calculate_scalar_value(const double t, const int src_point) {
  VectorXd u0 = VectorXd::Zero(V.rows());
  u0[0] = 1;
  auto start_time = chrono::system_clock::now();
  auto &&mid_tmp = MatrixXd(M-t*L);
  // the line below will call 'std::bad_alloc' when the mesh is too large
  VectorXd u = mid_tmp.householderQr().solve(u0);
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

  VectorXd &&u = calculate_scalar_value(atof(argv[2]), 0);

  cout << "scalar value calculation finished" << endl;

  VectorXd &&g = normalized_gradient(u);

  cout << "gradient normalization finished" << endl;

  // barycenter is a face_num * 3 matrix
  MatrixXd BC;
  barycenter(V, F, BC);
  MatrixXd T = MatrixXd::Zero(BC.rows(), BC.cols());
  for (int i = 0; i < F.rows(); ++i) {
    T(i, 0) = g[i+0*F.rows()];
    T(i, 1) = g[i+1*F.rows()];
    T(i, 2) = g[i+2*F.rows()];
  }
  const double length = avg_edge_length(V, F)*0.5;
  // write each gradeint vector to a file
  fstream out;
  out.open("out.txt", ios::out | ios::app);
  for (int i = 0; i < F.rows(); ++i) {
    // write format: start_x,start_y,start_z,end_x,end_y,end_z
    out << BC(i, 0) << "," << BC(i, 1) << "," << BC(i, 2) << "," <<
       BC(i, 0)+T(i, 0) << "," << BC(i, 1)+T(i, 1) << "," << BC(i, 2)+T(i, 2) << "\n";
  }
  out.flush();
  out.close();

  // calculate divergence of normalize gradient field

  // sort the divergence to get the desired verts

  // display the normalized gradient on screen
  // opengl::glfw::Viewer viewer;
  // viewer.data().set_mesh(V, F);
  // const RowVector3d color(255,0,0);
  // viewer.data().add_edges(BC, BC+length*T, color);
  // viewer.data().set_data(u);
  // // viewer.data().show_lines = false;
  // viewer.launch();
  return 0;
}