#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <filesystem>
#include <functional>
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
namespace fs = filesystem;

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

VectorXd calculate_divergence(VectorXd &g, const Model3D::CRichModel &model) {
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

  cout << "edge normal calculation finished" << endl;

  // compute divergence
  VectorXd div_u = VectorXd::Zero(model.GetNumOfVerts());
  for (int vi = 0; vi < model.GetNumOfVerts(); ++vi) {
    for (auto &&val : perp_vec_base[vi]) {
      // find the normalized gradient vector of corresponding face
      Vector3d ux {g[val.fi], g[val.fi+F.rows()], g[val.fi, 2*F.rows()]};
      div_u[vi] += 0.5 * model.Edge(val.ei).length * ux.dot(val.perp_vec);
    }
  }
  return div_u;
}

inline double variance(VectorXd &vec) {
  const double mean = vec.mean();
  double var = 0;
  for (int i = 0; i < vec.rows(); ++i) {
    var += pow(vec[i]-mean, 2);
  }
  return var / vec.rows();
}

void query_neighor_vertices(Model3D::CRichModel &model, const int root, function<void(int)> &&f) {
  auto &&neigh = model.Neigh(root);
  for (auto &&neigh_edge : neigh) {
    bool is_left = (model.Edge(neigh_edge.first).indexOfLeftVert == root);
    int neigh_vert;
    if (is_left) neigh_vert = model.Edge(neigh_edge.first).indexOfRightVert;
    else neigh_vert = model.Edge(neigh_edge.first).indexOfLeftVert;
    f(neigh_vert);
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "<bin> <filename>" << endl;
    exit(-1);
  }
  readOBJ(argv[1], V, F);
  // laplace cotan matrix
  cotmatrix(V, F, L);
  // mass mastrix for each triangle
  massmatrix(V, F, MASSMATRIX_TYPE_DEFAULT, M);
  // gradient operator
  grad(V, F, G);

  string scalar_cache_file = string(argv[1])+string(".scalar.txt");

  VectorXd u, g, div_u;

  // load cached scalar field if cache exists
  if (fs::exists(scalar_cache_file)) {
    cout << "using scalar cache:" << scalar_cache_file << endl;
    fstream scalar_in;
    scalar_in.open(scalar_cache_file, ios::in);
    string line;
    getline(scalar_in, line);
    int lines_num = stoi(line), index = 0;
    u.resize(lines_num);
    while (getline(scalar_in, line)) {
      u(index) = stod(line);
      index++;
    }
    cout << "scalar cache loaded" << endl;
    scalar_in.close();
  } else {
    cout << "start computing saclar field" << endl;
    const double t = pow(avg_edge_length(V, F), 2);
    u = calculate_scalar_value(t, 0);
    cout << "scalar field calculation finished" << endl;
    fstream scalar_out;
    scalar_out.open(scalar_cache_file, ios::app | ios::out);
    scalar_out << u.rows() << "\n";
    for (int i = 0; i < u.rows(); ++i) {
      scalar_out << u(i) << "\n";
    }
    scalar_out.flush();
    scalar_out.close();
    cout << "scalar cache generated" << endl;
  }
  g = normalized_gradient(u);

  cout << "gradient normalization finished" << endl;

  Model3D::CRichModel model(argv[1]);
  model.LoadModel();

  div_u = calculate_divergence(g, model);

  cout << "divergence calculation finished" << endl;

  // vertex texture coordinate
  VectorXd vt = div_u;

  // unify the divergence
  double min = vt[0], max = vt[0];
  for (int i = 0; i < vt.rows(); ++i) {
    min = min < vt[i]?min:vt[i];
    max = max > vt[i]?max:vt[i];
  }
  cout << "max:" << max << "\nmin:" << min << endl;
  double gap = abs(min), divider = max+abs(min);
  for (int i = 0; i < vt.rows(); ++i) {
    vt[i] += gap;
    vt[i] /= divider;
  }

  // write div_u for data analysis
  fstream data_out;
  fs::remove("div_u.data.txt");
  data_out.open("div_u.data.txt", ios::app | ios::out);
  for (int i = 0; i < V.rows(); ++i) {
    data_out << div_u[i] << "\n";
  }
  data_out.flush();
  data_out.close();

  // calculate the mean and variance of div_u
  const double mu = div_u.mean();
  const double var = variance(div_u);
  const double filter = mu-3*sqrt(var);
  set<int> base_collection;
  cout << "mean:" << mu << "\nvar:" << var << endl;
  // filter the vertices through mean and divergence
  for (int i = 0; i < V.rows(); ++i) {
    if (div_u[i] < filter) {
      vt[i] = 0;
      base_collection.insert(i);
    } else vt[i] = 0.5;
  }

  // if the degree of the vertex is 1 or 2
  // save it as a end point
  set<int> end_verts;
  set<int> iterate_collection = base_collection;
  for (auto &&base : iterate_collection) {
    int degree = 0;
    queue<int> neigh_in_set;
    query_neighor_vertices(model, base, [&](int neigh_vert) -> void {
      if (base_collection.find(neigh_vert) != base_collection.end()) {
        neigh_in_set.push(neigh_vert);
        degree++;
      }
    });
    if (degree == 1) {
      end_verts.insert(base);
    } else if (degree == 0) {
      // remove degree 0 vertices
      // cout << "erase vert due to zero degree:" << base << endl;
      base_collection.erase(base);
      // change the visual effect
      vt[base] = 0.5;
    } else if (degree == 2) {
      // query if the neighbors in set are in the same triangle
      // if so, set it as an end_point
      int neigh_vert_1 = neigh_in_set.front();
      neigh_in_set.pop();
      int neigh_vert_2 = neigh_in_set.front();
      int f1 = model.Edge(model.GetEdgeIndexFromTwoVertices(base, neigh_vert_1)).indexOfFrontFace;
      int f2 = model.Edge(model.GetEdgeIndexFromTwoVertices(base, neigh_vert_2)).indexOfFrontFace;
      int f3 = model.Edge(model.GetEdgeIndexFromTwoVertices(neigh_vert_1, base)).indexOfFrontFace;
      int f4 = model.Edge(model.GetEdgeIndexFromTwoVertices(neigh_vert_2, base)).indexOfFrontFace;
      if ((f1 == f4) || (f2 == f3)) {
        end_verts.insert(base);
        // cout << "insert degree two vert:" << base << endl;
      }
    }
  }

  cout << "end point num:" << end_verts.size() << endl;
  cout << "base point num:" << base_collection.size() << endl;

  // expand following the direction of the normalized gradient
  // calculate the gradient for eqach edge from the end_points
  for (auto &&end_vert : end_verts) {
    int cur = end_vert;
    while (true) {
      auto &&cur_vert = model.Vert(cur);
      double max_val = 0;
      int max_val_vert_index = -1;
      // process the neighobrs of vert cur
      query_neighor_vertices(model, cur, [&](int neigh_vert_index) -> void {
        // estimate the vector for the edge
        auto &&neigh_vert = model.Vert(neigh_vert_index);
        Vector3d edge_vec {neigh_vert.x-cur_vert.x,neigh_vert.y-cur_vert.y,neigh_vert.z-cur_vert.z};
        // estimate the scalar value for the edge
        int face_1 = model.Edge(model.GetEdgeIndexFromTwoVertices(cur, neigh_vert_index)).indexOfFrontFace;
        int face_2 = model.Edge(model.GetEdgeIndexFromTwoVertices(neigh_vert_index, cur)).indexOfFrontFace;
        Vector3d face_grad_1 {g[face_1], g[face_1+F.rows()], g[face_1+2*F.rows()]};
        Vector3d face_grad_2 {g[face_2], g[face_2+F.rows()], g[face_2+2*F.rows()]};
        const double new_val = edge_vec.dot(face_grad_1)+edge_vec.dot(face_grad_2);
        if (new_val > max_val) {
          max_val = new_val;
          max_val_vert_index = neigh_vert_index;
        }
      });
      // if the neighbor with the greatest value is in the base_collection, do nothing
      // otherwise insert it into the set
      // cout << max_val << endl;
      if (max_val_vert_index != -1) {
        if (base_collection.find(max_val_vert_index) == base_collection.end()) {
          // cout << max_val << ":insert new vertex:" << max_val_vert_index << endl;
          base_collection.insert(max_val_vert_index);
          vt[max_val_vert_index] = 1;
          // move a step forward
          cur = max_val_vert_index;
        } else break;
      } else break;
    }
  }

  string model_objfile_name = string(argv[1])+string(".divergence_method.obj");
  string model_mtlfile_name = string(argv[1])+string(".divergence_method.mtl");

  // write output model files
  fstream obj_out, mtl_out;
  fs::remove(model_objfile_name);
  fs::remove(model_mtlfile_name);
  obj_out.open(model_objfile_name, ios::app | ios::out);
  mtl_out.open(model_mtlfile_name, ios::app | ios::out);
  obj_out << "mtllib " << model_mtlfile_name << "\nusemtl Default\ng default\n";
  for (int i = 0;i < V.rows();++i) {
    obj_out << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
  }
  for (int i = 0;i < V.rows();++i) {
    obj_out << "vt " << vt(i) << " 0\n";
  }
  for (int i = 0;i < F.rows();++i) {
    obj_out <<"f "<<F(i, 0)+1<<"/"<<F(i, 0)+1<<" "<<F(i, 1)+1<<"/"<<F(i, 1)+1<<" "<<F(i, 2)+1<<"/"<<F(i, 2)+1<<"\n";
  }

  mtl_out << "newmtl Default\nKa 0 0 0\nKd 1 1 1\nKs 0.1 0.1 0.1\nmap_Kd colorbar.png\n";

  obj_out.flush();
  mtl_out.flush();
  obj_out.close();
  mtl_out.close();

  MatrixXd BC;
  barycenter(V, F, BC);
  MatrixXd T = MatrixXd::Zero(BC.rows(), BC.cols());
  for (int i = 0; i < F.rows(); ++i) {
    T(i, 0) = g[i+0*F.rows()];
    T(i, 1) = g[i+1*F.rows()];
    T(i, 2) = g[i+2*F.rows()];
  }
  const double length = avg_edge_length(V, F)*0.5;

  // visualize the data
  opengl::glfw::Viewer viewer;
  const RowVector3d color(255,0,0);
  viewer.data().set_mesh(V, F);
  viewer.data().add_edges(BC, BC+length*T, color);
  viewer.data().set_data(vt);
  viewer.launch();

  return 0;
}