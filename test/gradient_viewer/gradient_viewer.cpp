#include <iostream>
#include <fstream>
#include <string>
#include <igl/grad.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/readOBJ.h>
#include <igl/heat_geodesics.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <Eigen/QR>

using namespace std;
using namespace igl;
using namespace Eigen;

MatrixXd V;
MatrixXi F;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cout << "<bin> <filename> <t>" << endl;
    exit(-1);
  }
  readOBJ(argv[1], V, F);
  SparseMatrix<double> G, L, M;
  cotmatrix(V, F, L);
  massmatrix(V, F, MASSMATRIX_TYPE_DEFAULT, M);
  const double t = atof(argv[2]);
  VectorXd u0 = VectorXd::Zero(V.rows());
  u0[0] = 1;
  SparseMatrix<double> Q = M-t*L;
  SparseMatrix<double> _;
  min_quad_with_fixed_data<double> neumann;
  min_quad_with_fixed_precompute(Q, VectorXi(), _, true, neumann);
  VectorXd u;
  min_quad_with_fixed_solve(neumann, u0, VectorXd(), VectorXd(), u);
  // the sollowing code willl be optimized with libigl code
  // // the line below will call 'std::bad_alloc' when the mesh is too large
  // VectorXd u = MatrixXd(M-L*t).householderQr().solve(u0);
  cout << "calculation finished" << endl;
  grad(V, F, G);
  VectorXd g = G * u;
  auto m = F.rows();
  cout << "normalize gradient" << endl;
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
  MatrixXd BC;
  barycenter(V, F, BC);
  MatrixXd T = MatrixXd::Zero(BC.rows(), BC.cols());
  for (int i = 0; i < F.rows(); ++i) {
    T(i, 0) = g[i+0*F.rows()];
    T(i, 1) = g[i+1*F.rows()];
    T(i, 2) = g[i+2*F.rows()];
  }
  const double length = avg_edge_length(V, F)*0.5;
  // display the normalized gradient on screen
  fstream out;
  out.open("out.txt", ios::out | ios::app);
  for (int i = 0; i < F.rows(); ++i) {
    out << BC(i, 0) << "," << BC(i, 1) << "," << BC(i, 2) << "," <<
       BC(i, 0)+T(i, 0) << "," << BC(i, 1)+T(i, 1) << "," << BC(i, 2)+T(i, 2) << "\n";
  }
  out.flush();
  out.close();
  opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  const RowVector3d color(255,0,0);
  viewer.data().add_edges(BC, BC+length*T, color);
  viewer.data().set_data(u);
  // viewer.data().show_lines = false;
  viewer.launch();
  return 0;
}