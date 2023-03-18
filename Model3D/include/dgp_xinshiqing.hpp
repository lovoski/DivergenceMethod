#pragma once
#include <igl/cotmatrix.h>
#include <igl/bfs_orient.h>
#include <igl/centroid.h>
#include <igl/volume.h>
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/slice.h>
#include <igl/harmonic.h>
#include <igl/boundary_loop.h>
#include <igl/copyleft/cgal/convex_hull.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <CGAL/IO/OBJ.h>
#include <CGAL/Exact_integer.h>
#include <CGAL/Homogeneous.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Straight_skeleton_2.h>
#include <CGAL/Surface_mesh/Properties.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/IO/Nef_polyhedron_iostream_3.h>
#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include<CGAL/create_straight_skeleton_2.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <cgal/Polygon_mesh_processing/corefinement.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/point_generators_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/Fuzzy_iso_box.h>
#include <CGAL/Search_traits_2.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/graph_traits_Surface_mesh.h>
#include <CGAL/subdivision_method_3.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Timer.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Bounded_normal_change_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/GarlandHeckbert_policies.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/IO/Nef_polyhedron_iostream_3.h>
#include <CGAL/minkowski_sum_3.h>
#include <chrono>
#include <boost/lexical_cast.hpp>
#include <boost/iterator/function_output_iterator.hpp>
#include <iostream>
#include "UnionFind.hpp"
#include "CDT.h"
#include "BaseModel.h"
#include "RichModel.h"
#include "EdgePoint.h"
using namespace std;

namespace Model3D
{
	typedef CGAL::Exact_predicates_exact_constructions_kernel   exact_Kernel;
	typedef CGAL::Exact_predicates_inexact_constructions_kernel  inexact_Kernel;
	typedef CGAL::Cartesian<double> double_Kernel;
	//typedef Kernel::Point_3 Point_3;
	//typedef CGAL::Surface_mesh<Kernel::Point_3>                         Surface_mesh;
	//typedef CGAL::Polyhedron_3<Kernel>                          Polyhedron_3;
	//typedef CGAL::Nef_polyhedron_3<Kernel>                      Nef_polyhedron_3;
	//typedef CGAL::Plane_3<Kernel>      Plane_3;

	template<typename K>
	void TransformPolygon2Wall3D(const CGAL::Polygon_2<K>& poly, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
	{
		int n = poly.size();
		V.resize(2 * n, 3);
		for (int i = 0; i < n; ++i)
		{
			V(i, 0) = poly.vertex(i).x();
			V(i, 1) = poly.vertex(i).y();
			V(i, 2) = 0;
		}
		for (int i = 0; i < n; ++i)
		{
			double len1 = sqrt((poly.vertex(i) - poly.vertex((i + n - 1) % n)).squared_length());
			double len2 = sqrt((poly.vertex(i) - poly.vertex((i + 1) % n)).squared_length());
			V(n + i, 0) = poly.vertex(i).x();
			V(n + i, 1) = poly.vertex(i).y();
			V(n + i, 2) = (len1 + len2) / 2;
		}

		F.resize(2 * n, 3);
		for (int i = 0; i < n; ++i)
		{
			F(2 * i, 0) = i;
			F(2 * i, 1) = (i + 1) % n;
			F(2 * i, 2) = (i + 1) % n + n;
			F(2 * i + 1, 0) = i;
			F(2 * i + 1, 1) = (i + 1) % n + n;
			F(2 * i + 1, 2) = i + n;
		}

		////////{
		////////	//����TransformPolygon2Wall3D��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	typedef CGAL::Polygon_2< double_Kernel> Poly_2;
		////////	Poly_2 poly;
		////////	poly.push_back(double_Kernel::Point_2(0, 0));
		////////	poly.push_back(double_Kernel::Point_2(1, 0));
		////////	poly.push_back(double_Kernel::Point_2(1, 1));
		////////	poly.push_back(double_Kernel::Point_2(0, 1));
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	TransformPolygon2Wall3D(poly, V, F);
		////////	igl::writeOBJ("out.obj", V, F);
		////////}
	}

	template<typename Kernel>
	void ConvertSurfaceMesh2Model3D(const CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm,
		Model3D::CBaseModel& model)
	{
		typedef CGAL::Point_3<Kernel> Point_3;
		typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
		namespace PMP = CGAL::Polygon_mesh_processing;
		vector<Model3D::CPoint3D> verts;
		verts.resize(sm.num_vertices());
		for (int i = 0; i < sm.num_vertices(); ++i)
		{
			verts[i].x = sm.point(CGAL::SM_Vertex_index(i)).x();
			verts[i].y = sm.point(CGAL::SM_Vertex_index(i)).y();
			verts[i].z = sm.point(CGAL::SM_Vertex_index(i)).z();
		}

		vector<Model3D::CBaseModel::CFace> faces;
		faces.resize(sm.number_of_faces());
		int i = 0;
		for (auto face_index : sm.faces())
		{
			int j = 0;
			CGAL::Vertex_around_face_circulator<Surface_mesh> vcirc(sm.halfedge(face_index), sm), done(vcirc);
			do
			{
				faces[i][j] = *vcirc++;
				++j;
			} while (vcirc != done);
			++i;
		}
		model = Model3D::CBaseModel(verts, faces);

		////////{
		////////	//����ConvertSurfaceMesh2Model3D��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	using namespace Model3D;
		////////	Surface_mesh sm;
		////////	CGAL::IO::read_polygon_mesh("sphere.obj", sm);
		////////	CBaseModel model;
		////////	ConvertSurfaceMesh2Model3D(sm, model);
		////////	model.SaveObjFile("sphere_sm.obj");
		////////}
	}

	template<typename Kernel>
	void RepairSurfaceMesh(CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm)
	{
		namespace PMP = CGAL::Polygon_mesh_processing;
		typedef CGAL::Point_3<Kernel> Point_3;
		typedef CGAL::Surface_mesh<CGAL::Point_3<Kernel>> Surface_mesh;

		std::vector<Point_3 > points;
		std::vector<std::vector<int> > triangles;
		for (int i = 0; i < sm.num_vertices(); ++i)
		{
			points.emplace_back(sm.point(CGAL::SM_Vertex_index(i)));
		}
		for (auto face_index : sm.faces())
		{
			int j = 0;
			CGAL::Vertex_around_face_circulator<Surface_mesh> vcirc(sm.halfedge(face_index), sm), done(vcirc);
			std::vector<int> tri;
			do
			{
				tri.emplace_back(*vcirc++);
				++j;
			} while (vcirc != done);

			for (int j = 2; j < tri.size(); ++j)
			{
				std::vector<int> tri_sub;
				tri_sub.emplace_back(tri[0]);
				tri_sub.emplace_back(tri[j-1]);
				tri_sub.emplace_back(tri[j]);
				triangles.emplace_back(tri_sub);
			}
		}
		PMP::repair_polygon_soup(points, triangles);
		sm.clear();
		PMP::polygon_soup_to_polygon_mesh(points, triangles, sm);
	}

	template<typename Kernel>
	void ConvertSurfaceMesh2Matrix(const CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm,
		Eigen::MatrixXd& verts, Eigen::MatrixXi& faces)
	{
		typedef CGAL::Point_3<Kernel> Point_3;
		typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
		namespace PMP = CGAL::Polygon_mesh_processing;

		verts.resize(sm.num_vertices(), 3);
		for (int i = 0; i < sm.num_vertices(); ++i)
		{
			verts(i, 0) = sm.point(CGAL::SM_Vertex_index(i)).x();
			verts(i, 1) = sm.point(CGAL::SM_Vertex_index(i)).y();
			verts(i, 2) = sm.point(CGAL::SM_Vertex_index(i)).z();
		}

		faces.resize(sm.num_faces(), 3);
		int i = 0;
		for (auto face_index : sm.faces())
		{
			int j = 0;
			CGAL::Vertex_around_face_circulator<Surface_mesh> vcirc(sm.halfedge(face_index), sm), done(vcirc);
			do
			{
				faces(i, j) = *vcirc++;
				++j;
			} while (vcirc != done);
			++i;
		}

		////////{
		////////	//����ConvertSurfaceMesh2Model3D��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	using namespace Model3D;
		////////	Surface_mesh sm;
		////////	CGAL::IO::read_polygon_mesh("sphere.obj", sm);
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	ConvertSurfaceMesh2Matrix(sm, V, F);
		////////	igl::writeOBJ("sphere_out.obj", V, F);
		////////}
	}

	template<typename Kernel>
	void ConvertModel3D2SurfaceMesh(const  Model3D::CBaseModel& model,
		CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm)
	{
		typedef CGAL::Point_3<Kernel> Point_3;
		sm.clear();
		for (int i = 0; i < model.GetNumOfVerts(); ++i)
		{
			sm.add_vertex(Point_3(model.Vert(i).x, model.Vert(i).y, model.Vert(i).z));
		}
		for (int i = 0; i < model.GetNumOfFaces(); ++i)
		{
			sm.add_face(CGAL::SM_Vertex_index(model.Face(i)[0]),
				CGAL::SM_Vertex_index(model.Face(i)[1]),
				CGAL::SM_Vertex_index(model.Face(i)[2]));
		}

		////////{
		////////	//����ConvertModel3D2SurfaceMesh��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	using namespace Model3D;
		////////	CBaseModel model("sphere.obj");
		////////	model.LoadModel();
		////////	Surface_mesh sm;
		////////	ConvertModel3D2SurfaceMesh(model, sm);
		////////	CGAL::IO::write_polygon_mesh("sphere_out.obj", sm);
		////////}
	}

	void ConvertMatrix2Model3D(const  Eigen::MatrixXd& V, const  Eigen::MatrixXi& F,
		Model3D::CBaseModel& model)
	{
		vector<Model3D::CPoint3D> verts(V.rows());
		for (int i = 0; i < V.rows(); ++i)
		{
			Model3D::CPoint3D p(V(i, 0), V(i, 1), V(i, 2));
			verts[i] = p;
		}
		vector<Model3D::CBaseModel::CFace> faces(F.rows());
		for (int i = 0; i < F.rows(); ++i)
		{
			faces[i] = Model3D::CBaseModel::CFace(CGAL::SM_Vertex_index(F(i, 0)),
				CGAL::SM_Vertex_index(F(i, 1)),
				CGAL::SM_Vertex_index(F(i, 2)));
		}
		model = Model3D::CBaseModel(verts, faces);

		////////{
		////////	//����ConvertMatrix2Model3D��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	using namespace Model3D;
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	igl::readOBJ("sphere.obj", V, F);
		////////	CBaseModel model;
		////////	ConvertMatrix2Model3D(V, F, model);
		////////	model.SaveObjFile("sphere_out.obj");
		////////}
	}

	void ConvertModel3D2Matrix(const Model3D::CBaseModel& model,
		Eigen::MatrixXd& verts, Eigen::MatrixXi& faces)
	{
		verts.resize(model.GetNumOfVerts(), 3);
		for (int i = 0; i < model.GetNumOfVerts(); ++i)
		{
			verts(i, 0) = model.Vert(i).x;
			verts(i, 1) = model.Vert(i).y;
			verts(i, 2) = model.Vert(i).z;
		}

		faces.resize(model.GetNumOfFaces(), 3);
		for (int i = 0; i < model.GetNumOfFaces(); ++i)
		{
			for (int j = 0; j < 3; ++j)
				faces(i, j) = model.Face(i)[j];
		}

		////////{
		////////	//����ConvertModel3D2Matrix��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	using namespace Model3D;

		////////	CBaseModel model("sphere.obj");
		////////	model.LoadModel();
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	ConvertModel3D2Matrix(model, V, F);
		////////	igl::writeOBJ("sphere_out.obj", V, F);
		////////}
	}

	template<typename Kernel>
	void ConvertMatrix2SurfaceMesh(const  Eigen::MatrixXd& verts, const  Eigen::MatrixXi& faces,
		CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm)
	{
		typedef CGAL::Point_3<Kernel> Point_3;
		sm.clear();
		for (int i = 0; i < verts.rows(); ++i)
		{
			Point_3 p(verts(i, 0), verts(i, 1), verts(i, 2));
			sm.add_vertex(p);
		}
		for (int i = 0; i < faces.rows(); ++i)
		{
			sm.add_face(CGAL::SM_Vertex_index(faces(i, 0)),
				CGAL::SM_Vertex_index(faces(i, 1)),
				CGAL::SM_Vertex_index(faces(i, 2)));
		}

		////////{
		////////	//����ConvertMatrix2SurfaceMesh��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	using namespace Model3D;
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	igl::readOBJ("sphere.obj", V, F);
		////////	Surface_mesh sm;
		////////	ConvertMatrix2SurfaceMesh(V, F, sm);
		////////	CGAL::IO::write_polygon_mesh("sphere_out.obj", sm);
		////////}
	}

	template<typename Point_2>
	void ConvertPolygon2PlanarMesh(const vector<Point_2>& points,
		Eigen::MatrixXd& verts, Eigen::MatrixXi& faces)
	{
		typedef inexact_Kernel CDT_Kernel;
		CDT cdt;
		map<CDT_Kernel::Point_2, int> fromPoint2ID;
		vector<CDT_Kernel::Point_2> points_cdt;
		for (int i = 0; i < points.size(); ++i)
		{
			points_cdt.emplace_back(CDT_Kernel::Point_2(points[i].x(), points[i].y()));
			fromPoint2ID[points_cdt.back()] = points_cdt.size() - 1;
		}

		cdt.insert_constraint(points_cdt.begin(), points_cdt.end(), true);
		mark_domains(cdt);

		typedef Eigen::Vector3i FACE;
		vector<FACE> face_pool;
		for (CDT::Finite_faces_iterator fit = cdt.finite_faces_begin();
			fit != cdt.finite_faces_end(); ++fit)
		{
			if (fit->info().in_domain())
			{
				int id1 = fromPoint2ID[fit->vertex(0)->point()];
				int id2 = fromPoint2ID[fit->vertex(1)->point()];
				int id3 = fromPoint2ID[fit->vertex(2)->point()];
				face_pool.emplace_back(FACE(id1, id2, id3));
			}
		}

		verts.resize(points_cdt.size(), 3);
		for (int i = 0; i < points_cdt.size(); ++i)
		{
			verts(i, 0) = points_cdt[i].x();
			verts(i, 1) = points_cdt[i].y();
			verts(i, 2) = 0;
		}
		faces.resize(face_pool.size(), 3);
		for (int i = 0; i < face_pool.size(); ++i)
		{
			faces.row(i).array() = face_pool[i].array();
		}
		////////{
		////////	//����ConvertPolygon2PlanarMesh��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	typedef CGAL::Polygon_2<double_Kernel> Polygon_2;
		////////	using namespace Model3D;
		////////	Polygon_2 poly;
		////////	poly.push_back(double_Kernel::Point_2(0, 0));
		////////	poly.push_back(double_Kernel::Point_2(1, 0));
		////////	poly.push_back(double_Kernel::Point_2(1, 1));
		////////	poly.push_back(double_Kernel::Point_2(0, 1));
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	ConvertPolygon2PlanarMesh(poly.vertices(), V, F);
		////////	igl::writeOBJ("quad_out.obj", V, F);
		////////}
	}

	template<typename Point_2>
	void ConvertPolygon2PlanarMeshWithAddingSkeletalPoints(const vector<Point_2>& points,
		Eigen::MatrixXd& verts, Eigen::MatrixXi& faces)
	{
		typedef inexact_Kernel CDT_Kernel;
		CDT cdt;
		map<CDT_Kernel::Point_2, int> fromPoint2ID;
		vector<CDT_Kernel::Point_2> points_cdt;
		for (int i = 0; i < points.size(); ++i)
		{
			points_cdt.emplace_back(CDT_Kernel::Point_2(points[i].x(), points[i].y()));
			fromPoint2ID[points_cdt.back()] = points_cdt.size() - 1;
		}

		cdt.insert_constraint(points_cdt.begin(), points_cdt.end(), true);
		typedef CGAL::Straight_skeleton_2<CDT_Kernel> Ss;
		typedef boost::shared_ptr<Ss> SsPtr;
		SsPtr iss = CGAL::create_interior_straight_skeleton_2(points.begin(), points.end());
		for (auto i = iss->vertices_begin(); i != iss->vertices_end(); ++i)
		{
			if (i->is_contour())
				continue;
			cdt.insert(i->point());
			points_cdt.emplace_back(i->point());
			fromPoint2ID[i->point()] = points_cdt.size() - 1;
		}

		mark_domains(cdt);

		typedef Eigen::Vector3i FACE;
		vector<FACE> face_pool;
		for (CDT::Finite_faces_iterator fit = cdt.finite_faces_begin();
			fit != cdt.finite_faces_end(); ++fit)
		{
			if (fit->info().in_domain())
			{
				int id1 = fromPoint2ID[fit->vertex(0)->point()];
				int id2 = fromPoint2ID[fit->vertex(1)->point()];
				int id3 = fromPoint2ID[fit->vertex(2)->point()];
				face_pool.emplace_back(FACE(id1, id2, id3));
			}
		}

		verts.resize(points_cdt.size(), 3);
		for (int i = 0; i < points_cdt.size(); ++i)
		{
			verts(i, 0) = points_cdt[i].x();
			verts(i, 1) = points_cdt[i].y();
			verts(i, 2) = 0;
		}
		faces.resize(face_pool.size(), 3);
		for (int i = 0; i < face_pool.size(); ++i)
		{
			faces.row(i).array() = face_pool[i].array();
		}
		////////{
		////////	//����ConvertPolygon2PlanarMeshWithAddingSkeletalPoints��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	typedef CGAL::Polygon_2<double_Kernel> Polygon_2;
		////////	using namespace Model3D;
		////////	Polygon_2 poly;
		////////	poly.push_back(double_Kernel::Point_2(0, 0));
		////////	poly.push_back(double_Kernel::Point_2(1, 0));
		////////	poly.push_back(double_Kernel::Point_2(1, 1));
		////////	poly.push_back(double_Kernel::Point_2(0, 1));
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	ConvertPolygon2PlanarMesh(poly.vertices(), V, F);
		////////	igl::writeOBJ("quad_out.obj", V, F);
		////////}
	}

	template<class Kernel>
	void ConvertMatrix2Polyhedron(const  Eigen::MatrixXd& verts, const  Eigen::MatrixXi& faces,
		CGAL::Polyhedron_3<Kernel>& out)
	{
		out.clear();
		std::stringstream off_string;
		off_string << "OFF" << "\n";
		off_string << verts.rows() <<
			" " << faces.rows() << " " << 0 << "\n";
		for (int i = 0; i < verts.rows(); ++i)
			off_string << verts(i, 0) << " " << verts(i, 1) << " "
			<< verts(i, 2) << "\n";
		for (int i = 0; i < faces.rows(); ++i)
			off_string << "3 " << faces(i, 0)
			<< " " << faces(i, 1)
			<< " " << faces(i, 2) << "\n";
		off_string >> out;

		////////{
		////////	//����ConvertMatrix2Polyhedron��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	typedef CGAL::Polygon_2<double_Kernel> Polygon_2;
		////////	typedef CGAL::Polyhedron_3<double_Kernel> Polyhedron_3;
		////////	using namespace Model3D;
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	igl::readOBJ("sphere.obj", V, F);
		////////	Polyhedron_3 poly;
		////////	ConvertMatrix2Polyhedron(V, F, poly);
		////////	ofstream("sphere_out.off") << poly;
		////////}
	}

	template<typename Kernel>
	CGAL::AABB_tree<CGAL::AABB_traits<Kernel, CGAL::AABB_face_graph_triangle_primitive<CGAL::Surface_mesh<CGAL::Point_3<Kernel>>>>> GetAABBTree(const CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm)
	{
		typedef CGAL::AABB_tree<CGAL::AABB_traits<Kernel, CGAL::AABB_face_graph_triangle_primitive<CGAL::Surface_mesh<CGAL::Point_3<Kernel>>>>> Tree;
		return Tree(sm.faces_begin(), sm.faces_end(), sm);
	}

	template<typename Kernel>
	void MyBooleanDifference(const CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm1,
		const CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm2,
		CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm_out)
	{
		namespace PMP = CGAL::Polygon_mesh_processing;
		typedef CGAL::Point_3<Kernel> Point_3;
		typedef CGAL::Surface_mesh<CGAL::Point_3<Kernel>> Surface_mesh;
		typedef CGAL::AABB_tree<CGAL::AABB_traits<Kernel, CGAL::AABB_face_graph_triangle_primitive<CGAL::Surface_mesh<CGAL::Point_3<Kernel>>>>> Tree;
		Surface_mesh sm1_cpy = sm1;
		Surface_mesh sm2_cpy = sm2;
		if (is_closed(sm1) && is_closed(sm2))
		{
			PMP::corefine_and_compute_difference(sm1_cpy, sm2_cpy, sm_out);
			return;
		}

		PMP::corefine(sm1_cpy, sm2_cpy);

		std::vector<Point_3 > points;
		std::vector<std::vector<int> > triangles;
		Tree tree1 = GetAABBTree(sm1_cpy);
		Tree tree2 = GetAABBTree(sm2_cpy);

		typedef Surface_mesh::Face_index SM_Face_index;
		auto GetFaceNormal = [](const Surface_mesh& sm, SM_Face_index face_index)
		{
			auto GetNormal = [](Point_3 p1, Point_3 p2, Point_3 p3)
			{
				return CGAL::cross_product(p2 - p1, p3 - p2);
			};
			Point_3 pts[3];
			int j = 0;
			CGAL::Vertex_around_face_circulator<Surface_mesh> vcirc(sm.halfedge(face_index), sm), done(vcirc);
			do
			{
				pts[j] = sm.point(*vcirc++);
				++j;
			} while (vcirc != done);
			return GetNormal(pts[0], pts[1], pts[2]);
		};

		for (auto face_index : sm1_cpy.faces())
		{
			Point_3 pts[3];
			int j = 0;
			CGAL::Vertex_around_face_circulator<Surface_mesh> vcirc(sm1_cpy.halfedge(face_index), sm1_cpy), done(vcirc);
			do
			{
				pts[j] = sm1_cpy.point(*vcirc++);
				++j;
			} while (vcirc != done);
			auto centroid = CGAL::centroid(pts[0], pts[1], pts[2]);
			auto proj = tree2.closest_point_and_primitive(centroid);
			auto faceNormal = GetFaceNormal(sm2_cpy, proj.second);
			if (CGAL::scalar_product(faceNormal, centroid - proj.first) > 0)
			{
				points.emplace_back(pts[0]);
				points.emplace_back(pts[1]);
				points.emplace_back(pts[2]);
				std::vector<int> ids;
				ids.emplace_back((int)points.size() - 3);
				ids.emplace_back((int)points.size() - 2);
				ids.emplace_back((int)points.size() - 1);
				triangles.emplace_back(ids);
			}
		}

		for (auto face_index : sm2_cpy.faces())
		{
			Point_3 pts[3];
			int j = 0;
			CGAL::Vertex_around_face_circulator<Surface_mesh> vcirc(sm2_cpy.halfedge(face_index), sm2_cpy), done(vcirc);
			do
			{
				pts[j] = sm2_cpy.point(*vcirc++);
				++j;
			} while (vcirc != done);
			auto centroid = CGAL::centroid(pts[0], pts[1], pts[2]);
			auto proj = tree1.closest_point_and_primitive(centroid);
			auto faceNormal = GetFaceNormal(sm1_cpy, proj.second);
			if (CGAL::scalar_product(faceNormal, centroid - proj.first) < 0)
			{
				points.emplace_back(pts[0]);
				points.emplace_back(pts[1]);
				points.emplace_back(pts[2]);
				std::vector<int> ids;
				ids.emplace_back((int)points.size() - 2);
				ids.emplace_back((int)points.size() - 3);
				ids.emplace_back((int)points.size() - 1);
				triangles.emplace_back(ids);
			}
		}

		PMP::repair_polygon_soup(points, triangles);
		sm_out.clear();
		PMP::polygon_soup_to_polygon_mesh(points, triangles, sm_out);

		////////{
////////	Surface_mesh sm1, sm2;
////////	CGAL::IO::read_polygon_mesh("sphere.obj", sm1);
////////	MakeUnitSquare(sm2);
////////	Remesh(sm2, 0.02, 3, sm2);
////////	Scale(10, sm2);
////////	CGAL::IO::write_polygon_mesh("res2.obj", sm2);
////////	MyBooleanIntersection(sm1, sm2, sm_out);
////////}
	}

	void MakeSurfaceHarmonic(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
	{
		Eigen::VectorXi L;
		igl::boundary_loop(F, L);
		Eigen::MatrixXd initial(0, 3);
		for (int i = 0; i < L.size(); ++i)
		{
			initial.conservativeResize(initial.rows() + 1, 3);
			initial.row(i).array() = V.row(L(i)).array();
		}
		Eigen::MatrixXd out;
		igl::harmonic(V, F, L, initial, 1, out);
		V = out;
		{
			//////////////igl::read_triangle_mesh("cap.obj", V, F);
			//////////////MakeSurfaceHarmonic(V, F);
			//////////////igl::write_triangle_mesh("cap2.obj", V, F);
		}
	}

	CRichModel MakeOpenSurfaceTwoLayers(const CRichModel& input, double epsilon)
	{
		auto verts_new = input.m_Verts;
		auto faces_new = input.m_Faces;
		for (int i = 0; i < input.GetNumOfVerts(); ++i)
		{
			verts_new.emplace_back(input.Vert(i) - epsilon * input.Normal(i));
		}
		for (int i = 0; i < input.GetNumOfFaces(); ++i)
		{
			faces_new.emplace_back(CBaseModel::CFace(input.Face(i)[0] + input.GetNumOfVerts(),
				input.Face(i)[2] + input.GetNumOfVerts(),
				input.Face(i)[1] + input.GetNumOfVerts()));
		}
		for (int i = 0; i < input.GetNumOfEdges(); ++i)
		{
			if (input.IsExtremeEdge(i))
			{
				int left = input.Edge(i).indexOfLeftVert;
				int right = input.Edge(i).indexOfRightVert;
				faces_new.emplace_back(CBaseModel::CFace(left,
					right,
					right + input.GetNumOfVerts()));
				faces_new.emplace_back(CBaseModel::CFace(left,
					right + input.GetNumOfVerts(),
					left + input.GetNumOfVerts()));
			}
		}
		return CRichModel(verts_new, faces_new);
		////////{
		////////	MakeUnitSquare(sm);
		////////	CBaseModel model;
		////////	ConvertSurfaceMesh2Model3D(sm, model);
		////////	auto model2 = MakeOpenSurfaceTwoLayers(CRichModel(model), 0.1);
		////////	model2.SaveObjFile("closed.obj");
		////////}
	}

	template<typename Kernel>
	void MyBooleanIntersection(const CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm1,
		const CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm2,
		CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm_out)
	{
		namespace PMP = CGAL::Polygon_mesh_processing;
		typedef CGAL::Point_3<Kernel> Point_3;
		typedef CGAL::Surface_mesh<CGAL::Point_3<Kernel>> Surface_mesh;
		typedef CGAL::AABB_tree<CGAL::AABB_traits<Kernel, CGAL::AABB_face_graph_triangle_primitive<CGAL::Surface_mesh<CGAL::Point_3<Kernel>>>>> Tree;

		Surface_mesh sm1_cpy = sm1;
		Surface_mesh sm2_cpy = sm2;
		if (is_closed(sm1) && is_closed(sm2))
		{
			PMP::corefine_and_compute_intersection(sm1_cpy, sm2_cpy, sm_out);
			return;
		}


		PMP::corefine(sm1_cpy, sm2_cpy);

		std::vector<Point_3 > points;
		std::vector<std::vector<int> > triangles;
		Tree tree1 = GetAABBTree(sm1_cpy);
		Tree tree2 = GetAABBTree(sm2_cpy);

		typedef Surface_mesh::Face_index SM_Face_index;
		auto GetFaceNormal = [](const Surface_mesh& sm, SM_Face_index face_index)
		{
			auto GetNormal = [](Point_3 p1, Point_3 p2, Point_3 p3)
			{
				return CGAL::cross_product(p2 - p1, p3 - p2);
			};
			Point_3 pts[3];
			int j = 0;
			CGAL::Vertex_around_face_circulator<Surface_mesh> vcirc(sm.halfedge(face_index), sm), done(vcirc);
			do
			{
				pts[j] = sm.point(*vcirc++);
				++j;
			} while (vcirc != done);
			return GetNormal(pts[0], pts[1], pts[2]);
		};

		for (auto face_index : sm1_cpy.faces())
		{
			Point_3 pts[3];
			int j = 0;
			CGAL::Vertex_around_face_circulator<Surface_mesh> vcirc(sm1_cpy.halfedge(face_index), sm1_cpy), done(vcirc);
			do
			{
				pts[j] = sm1_cpy.point(*vcirc++);
				++j;
			} while (vcirc != done);
			auto centroid = CGAL::centroid(pts[0], pts[1], pts[2]);
			auto proj = tree2.closest_point_and_primitive(centroid);
			auto faceNormal = GetFaceNormal(sm2_cpy, proj.second);
			if (CGAL::scalar_product(faceNormal, centroid - proj.first) < 0)
			{
				points.emplace_back(pts[0]);
				points.emplace_back(pts[1]);
				points.emplace_back(pts[2]);
				std::vector<int> ids;
				ids.emplace_back((int)points.size() - 3);
				ids.emplace_back((int)points.size() - 2);
				ids.emplace_back((int)points.size() - 1);
				triangles.emplace_back(ids);
			}
		}

		for (auto face_index : sm2_cpy.faces())
		{
			Point_3 pts[3];
			int j = 0;
			CGAL::Vertex_around_face_circulator<Surface_mesh> vcirc(sm2_cpy.halfedge(face_index), sm2_cpy), done(vcirc);
			do
			{
				pts[j] = sm2_cpy.point(*vcirc++);
				++j;
			} while (vcirc != done);
			auto centroid = CGAL::centroid(pts[0], pts[1], pts[2]);
			auto proj = tree1.closest_point_and_primitive(centroid);
			auto faceNormal = GetFaceNormal(sm1_cpy, proj.second);
			if (CGAL::scalar_product(faceNormal, centroid - proj.first) < 0)
			{
				points.emplace_back(pts[0]);
				points.emplace_back(pts[1]);
				points.emplace_back(pts[2]);
				std::vector<int> ids;
				ids.emplace_back((int)points.size() - 3);
				ids.emplace_back((int)points.size() - 2);
				ids.emplace_back((int)points.size() - 1);
				triangles.emplace_back(ids);
			}
		}

		PMP::repair_polygon_soup(points, triangles);
		sm_out.clear();
		PMP::polygon_soup_to_polygon_mesh(points, triangles, sm_out);

		////////{
		////////	Surface_mesh sm1, sm2;
		////////	CGAL::IO::read_polygon_mesh("sphere.obj", sm1);
		////////	MakeUnitSquare(sm2);
		////////	Remesh(sm2, 0.02, 3, sm2);
		////////	Scale(10, sm2);
		////////	CGAL::IO::write_polygon_mesh("res2.obj", sm2);
		////////	MyBooleanIntersection(sm1, sm2, sm_out);
		////////}
	}

	template<typename Kernel>
	void Scale(CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm, double scale)
	{
		auto oldCentroid = CGAL::centroid(sm.points().begin(), sm.points().end());
		for (auto vi : sm.vertices())
		{
			auto diff = sm.point(vi) - oldCentroid;
			sm.point(vi) = oldCentroid + diff * scale;
		}
	}

	void NormalScale2Unit(Eigen::MatrixXd& V)
	{
		auto oldCentroid = V.colwise().sum() / V.rows();
		auto minCorner = (V.rowwise() - oldCentroid).colwise().minCoeff();
		auto maxCorner = (V.rowwise() - oldCentroid).colwise().maxCoeff();
		double scale = 2.0 / (maxCorner - minCorner).maxCoeff();
		for (int i = 0; i < V.rows(); ++i)
		{
			auto newPt = scale * (V.row(i).array() - oldCentroid.array());
			V.row(i).array() = newPt;
		}
	}

	void Scale(Eigen::MatrixXd& V, double scale)
	{
		auto oldCentroid = V.colwise().sum() / V.rows();

		for (int i = 0; i < V.rows(); ++i)
		{
			V.row(i).array() = oldCentroid + scale * (V.row(i) - oldCentroid);
		}
	}

	template<typename Kernel>
	void MakeUnitSquare(CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm)
	{
		typedef CGAL::Point_3<Kernel> Point_3;
		sm.clear();
		CGAL::SM_Vertex_index id0 = sm.add_vertex(Point_3(0, 0, 0));
		CGAL::SM_Vertex_index id1 = sm.add_vertex(Point_3(1, 0, 0));
		CGAL::SM_Vertex_index id2 = sm.add_vertex(Point_3(1, 1, 0));
		CGAL::SM_Vertex_index id3 = sm.add_vertex(Point_3(0, 1, 0));
		sm.add_face(id0, id1, id2);
		sm.add_face(id0, id2, id3);
	}

	void MakeMobiusStrip(Eigen::MatrixXd& verts, Eigen::MatrixXi& faces, double R, double w, int segNumMajor, int segNumMinor)
	{
		verts.resize(0, 3);
		faces.resize(0, 3);
		double angleStp = 2 * M_PI / (double)segNumMajor;
		double stepWidth = 2 * w / (double)segNumMinor;
		map<pair<int, int>, int> idpair2id;

		double theta = 0;
		for (int i = 0; i <= segNumMajor; ++i, theta += angleStp)
		{
			double width = -w;
			double theta_2 = 0.5 * theta;
			for (int j = 0; j <= segNumMinor; ++j, width += stepWidth)
			{
				verts.conservativeResize(verts.rows() + 1, 3);
				verts.row(verts.rows() - 1).array() = Eigen::RowVector3d(
					(R + width * cos(theta_2)) * cos(theta),
					(R + width * cos(theta_2)) * sin(theta),
					width * sin(theta_2));
				idpair2id[make_pair(i, j)] = verts.rows() - 1;
			}
		}
		for (int i = 0; i < segNumMajor; ++i)
		{
			int nxtI = i + 1;
			for (int j = 0; j < segNumMinor; ++j)
			{
				int nxtJ = j + 1;
				faces.conservativeResize(faces.rows() + 1, 3);
				faces.row(faces.rows() - 1).array() = Eigen::RowVector3i(idpair2id[make_pair(i, j)],
					idpair2id[make_pair(nxtI, j)],
					idpair2id[make_pair(nxtI, nxtJ)]);
				faces.conservativeResize(faces.rows() + 1, 3);
				faces.row(faces.rows() - 1).array() = Eigen::RowVector3i(idpair2id[make_pair(i, j)],
					idpair2id[make_pair(nxtI, nxtJ)],
					idpair2id[make_pair(i, nxtJ)]);
			}
		}
	}

	void MakeTorus(Eigen::MatrixXd& verts, Eigen::MatrixXi& faces, double R, double r, int segNumMajor, int segNumMinor)
	{
		verts.resize(0, 3);
		faces.resize(0, 3);
		double angleStpR = 2 * M_PI / (double)segNumMajor;
		double angleStpr = 2 * M_PI / (double)segNumMinor;
		map<pair<int, int>, int> idpair2id;
		/*	x = (c + acosv)cosu
			(2)
				y = (c + acosv)sinu
				(3)
				z = asinv*/

		double thetaMajor = 0;
		for (int i = 0; i < segNumMajor; ++i, thetaMajor += angleStpR)
		{
			double thetaMinor = 0;
			for (int j = 0; j < segNumMinor; ++j, thetaMinor += angleStpr)
			{
				verts.conservativeResize(verts.rows() + 1, 3);
				verts.row(verts.rows() - 1).array() = Eigen::RowVector3d((R + r * cos(thetaMinor)) * cos(thetaMajor),
					(R + r * cos(thetaMinor)) * sin(thetaMajor), r * sin(thetaMinor));
				idpair2id[make_pair(i, j)] = verts.rows() - 1;
			}
		}
		for (int i = 0; i < segNumMajor; ++i)
		{
			int nxtI = (i + 1) % segNumMajor;
			for (int j = 0; j < segNumMinor; ++j)
			{
				int nxtJ = (j + 1) % segNumMinor;
				faces.conservativeResize(faces.rows() + 1, 3);
				faces.row(faces.rows() - 1).array() = Eigen::RowVector3i(idpair2id[make_pair(i, j)],
					idpair2id[make_pair(nxtI, j)],
					idpair2id[make_pair(nxtI, nxtJ)]);
				faces.conservativeResize(faces.rows() + 1, 3);
				faces.row(faces.rows() - 1).array() = Eigen::RowVector3i(idpair2id[make_pair(i, j)],
					idpair2id[make_pair(nxtI, nxtJ)],
					idpair2id[make_pair(i, nxtJ)]);
			}
		}
	}

	void MakeTorusKnot(Eigen::MatrixXd& verts, Eigen::MatrixXi& faces, double R, double r, int p, int q, int segNumMajor, int segNumMinor)
	{
		//% https://zh.wikipedia.org/zh-cn/%E7%8E%AF%E9%9D%A2%E7%BA%BD%E7%BB%93
		//p = 2;
		//q = 3;
		//t = 0:0.02 : (2 * pi);
		//x = cos(p * t).*(2 + cos(q * t));
		//y = sin(p * t).*(2 + cos(q * t));
		//z = sin(q * t);
		//plot3(x, y, z);
		//hold on;
		//axis equal
		verts.resize(0, 3);
		faces.resize(0, 3);
		double angleStpR = 2 * M_PI / (double)segNumMajor;
		double angleStpr = 2 * M_PI / (double)segNumMinor;
		map<pair<int, int>, int> idpair2id;
		//p = 2;
		//q = 3;
		//t = 0:0.02 : (2 * pi);
		//x = cos(p * t).*(2 + cos(q * t));
		//y = sin(p * t).*(2 + cos(q * t));
		//z = sin(q * t);
		//plot3(x, y, z);
		//hold on;
		//axis equal
		double thetaMajor = 0;
		for (int i = 0; i < segNumMajor; ++i, thetaMajor += angleStpR)
		{
			double thetaMinor = 0;
			for (int j = 0; j < segNumMinor; ++j, thetaMinor += angleStpr)
			{
				Eigen::RowVector3d basePnt(R * cos(p * thetaMajor) * (2 + cos(q * thetaMajor)),
					R * sin(p * thetaMajor) * (2 + cos(q * thetaMajor)),
					R * sin(q * thetaMajor));
				Eigen::RowVector3d forward(-p * sin(p * thetaMajor) * (2 + cos(q * thetaMajor)) + cos(p * thetaMajor) * (-q * sin(q * thetaMajor)),
					p * cos(p * thetaMajor) * (2 + cos(q * thetaMajor)) + sin(p * thetaMajor) * (-q * sin(q * thetaMajor)),
					q * cos(q * thetaMajor));
				forward.normalize();
				Eigen::RowVector3d z(0, 0, 1);
				Eigen::RowVector3d dir1 = z.cross(forward);
				dir1.normalize();
				Eigen::RowVector3d dir2 = forward.cross(dir1);
				dir2.normalize();
				Eigen::RowVector3d pt = basePnt + r * cos(thetaMinor) * dir2 + r * sin(thetaMinor) * dir1;
				verts.conservativeResize(verts.rows() + 1, 3);
				verts.row(verts.rows() - 1).array() = pt;
				idpair2id[make_pair(i, j)] = verts.rows() - 1;
			}
		}
		for (int i = 0; i < segNumMajor; ++i)
		{
			int nxtI = (i + 1) % segNumMajor;
			for (int j = 0; j < segNumMinor; ++j)
			{
				int nxtJ = (j + 1) % segNumMinor;
				faces.conservativeResize(faces.rows() + 1, 3);
				faces.row(faces.rows() - 1).array() = Eigen::RowVector3i(idpair2id[make_pair(i, j)],
					idpair2id[make_pair(nxtI, j)],
					idpair2id[make_pair(nxtI, nxtJ)]);
				faces.conservativeResize(faces.rows() + 1, 3);
				faces.row(faces.rows() - 1).array() = Eigen::RowVector3i(idpair2id[make_pair(i, j)],
					idpair2id[make_pair(nxtI, nxtJ)],
					idpair2id[make_pair(i, nxtJ)]);
			}
		}

		////////////{
		////////////	MakeTorusKnot(V, F, 1.5, 1, 2, 3, 300, 40);
		////////////	igl::write_triangle_mesh("knot.obj", V, F);
		////////////}
	}

	void MakeSphere(Eigen::MatrixXd& verts, Eigen::MatrixXi& faces, double r, int ptNum)
	{
		verts.resize(0, 3);
		double phi = M_PI * (3. - sqrt(5.));
		for (int i = 0; i < ptNum; ++i)
		{
			double h = 1 - i / (double)(ptNum - 1) * 2;
			double radius;
			if (i == 0 || i == ptNum - 1)
				radius = 0;
			else
				radius = sqrt(1 - h * h);
			double theta = phi * i;
			verts.conservativeResize(verts.rows() + 1, 3);
			verts.row(verts.rows() - 1).array() = Eigen::RowVector3d(r * cos(theta) * radius, r * h, r * sin(theta) * radius);
		}
		igl::copyleft::cgal::convex_hull(verts, faces);
	}

	void MakeCylinder(Eigen::MatrixXd& verts, Eigen::MatrixXi& faces, double r, double h, int segNum)
	{
		verts.resize(0, 3);
		faces.resize(0, 3);
		verts.conservativeResize(verts.rows() + 1, 3);
		verts.row(verts.rows() - 1).array() = Eigen::RowVector3d(0, 0, 0);
		double angleStp = 2 * M_PI / (double)segNum;
		double theta = 0;
		for (int i = 0; i < segNum; ++i, theta += angleStp)
		{
			verts.conservativeResize(verts.rows() + 1, 3);
			verts.row(verts.rows() - 1).array() = Eigen::RowVector3d(r * cos(theta), r * sin(theta), 0);
		}
		for (int i = 0; i < segNum; ++i, theta += angleStp)
		{
			verts.conservativeResize(verts.rows() + 1, 3);
			verts.row(verts.rows() - 1).array() = Eigen::RowVector3d(r * cos(theta), r * sin(theta), h);
		}
		verts.conservativeResize(verts.rows() + 1, 3);
		verts.row(verts.rows() - 1).array() = Eigen::RowVector3d(0, 0, h);
		int ptNumBottom = (verts.rows() - 2) / 2;
		for (int i = 1; i <= ptNumBottom; ++i)
		{
			int nxt = (i + 1);
			if (nxt > ptNumBottom)
				nxt = 1;
			faces.conservativeResize(faces.rows() + 1, 3);
			faces.row(faces.rows() - 1).array() = Eigen::RowVector3i(nxt, i, 0);
		}
		for (int i = 1; i <= ptNumBottom; ++i)
		{
			int nxt = (i + 1);
			if (nxt > ptNumBottom)
				nxt = 1;
			faces.conservativeResize(faces.rows() + 1, 3);
			faces.row(faces.rows() - 1).array() = Eigen::RowVector3i(i + ptNumBottom, nxt + ptNumBottom, verts.rows() - 1);
		}
		for (int i = 1; i <= ptNumBottom; ++i)
		{
			int nxt = (i + 1);
			if (nxt > ptNumBottom)
				nxt = 1;
			faces.conservativeResize(faces.rows() + 1, 3);
			faces.row(faces.rows() - 1).array() = Eigen::RowVector3i(i, nxt, i + ptNumBottom);
		}
		for (int i = 1; i <= ptNumBottom; ++i)
		{
			int nxt = (i + 1);
			if (nxt > ptNumBottom)
				nxt = 1;
			faces.conservativeResize(faces.rows() + 1, 3);
			faces.row(faces.rows() - 1).array() = Eigen::RowVector3i(nxt, nxt + ptNumBottom, i + ptNumBottom);
		}
	}

	void MakeCone(Eigen::MatrixXd& verts, Eigen::MatrixXi& faces, double r, double h, int segNum)
	{
		verts.resize(0, 3);
		faces.resize(0, 3);
		verts.conservativeResize(verts.rows() + 1, 3);
		verts.row(verts.rows() - 1).array() = Eigen::RowVector3d(0, 0, h);
		double angleStp = 2 * M_PI / (double)segNum;
		double theta = 0;
		for (int i = 0; i < segNum; ++i, theta += angleStp)
		{
			verts.conservativeResize(verts.rows() + 1, 3);
			verts.row(verts.rows() - 1).array() = Eigen::RowVector3d(r * cos(theta), r * sin(theta), 0);
		}
		verts.conservativeResize(verts.rows() + 1, 3);
		verts.row(verts.rows() - 1).array() = Eigen::RowVector3d(0, 0, 0);
		int ptNumBottom = verts.rows() - 2;
		for (int i = 1; i <= ptNumBottom; ++i)
		{
			int nxt = (i + 1);
			if (nxt > ptNumBottom)
				nxt = 1;
			faces.conservativeResize(faces.rows() + 1, 3);
			faces.row(faces.rows() - 1).array() = Eigen::RowVector3i(nxt, i, verts.rows() - 1);
		}
		for (int i = 1; i <= ptNumBottom; ++i)
		{
			int nxt = (i + 1);
			if (nxt > ptNumBottom)
				nxt = 1;
			faces.conservativeResize(faces.rows() + 1, 3);
			faces.row(faces.rows() - 1).array() = Eigen::RowVector3i(i, nxt, 0);
		}
	}
	void MakeUnitCube(Eigen::MatrixXd& verts, Eigen::MatrixXi& faces)
	{
		verts.resize(8, 3);
		verts <<
			0, 0, 0,
			0, 1, 0,
			1, 1, 0,
			1, 0, 0,
			0, 0, 1,
			0, 1, 1,
			1, 1, 1,
			1, 0, 1;
		faces.resize(12, 3);
		faces <<
			0, 1, 3,
			3, 1, 2,
			0, 4, 1,
			1, 4, 5,
			3, 2, 7,
			7, 2, 6,
			4, 0, 3,
			7, 4, 3,
			6, 4, 7,
			6, 5, 4,
			1, 5, 6,
			2, 1, 6;

		////////{
		////////	//����MakeUnitCube��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	typedef CGAL::Polygon_2<double_Kernel> Polygon_2;
		////////	typedef CGAL::Polyhedron_3<double_Kernel> Polyhedron_3;
		////////	using namespace Model3D;
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	MakeUnitCube(V, F);
		////////	igl::writeOBJ("cube.obj", V, F);
		////////}
	}

	template<typename Kernel>
	void MakeLargePlate(CGAL::Point_3<Kernel> center,
		CGAL::Vector_3<Kernel> dir1, double len1,
		CGAL::Vector_3<Kernel> dir2, double len2,
		CGAL::Vector_3<Kernel> dir3, double len3,
		CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm)
	{
		namespace PMP = CGAL::Polygon_mesh_processing;
		sm.clear();
		Eigen::MatrixXd verts;
		Eigen::MatrixXi faces;
		MakeUnitCube(verts, faces);
		ConvertMatrix2SurfaceMesh(verts, faces, sm);
		auto oldCentroid = PMP::centroid(sm);
		dir1 = dir1 / sqrt(dir1.squared_length());
		dir2 = dir2 / sqrt(dir2.squared_length());
		dir3 = dir3 / sqrt(dir3.squared_length());
		for (auto vi : sm.vertices())
		{
			auto diff = sm.point(vi) - oldCentroid;
			sm.point(vi) = center + diff.x() * len1 * dir1 + diff.y() * len2 * dir2 + diff.z() * len3 * dir3;
		}
		////////{
		////////	Surface_mesh input;
		////////	CGAL::IO::read_polygon_mesh("sphere.obj", input);
		////////	MakeLargePlate(0.1, inexact_Kernel::Point_3(0, 0, 0), inexact_Kernel::Vector_3(1, 1, 1), 100, sm);
		////////	PMP::corefine_and_compute_difference(input, sm, sm_out);
		////////	CGAL::IO::write_polygon_mesh("res.obj", sm_out);
		////////}
	}

	template<typename Kernel>
	void MakeLargePlate(double thickness, CGAL::Point_3<Kernel> center, CGAL::Vector_3<Kernel> dir, double largeSideLen,
		CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm)
	{
		namespace PMP = CGAL::Polygon_mesh_processing;
		sm.clear();
		Eigen::MatrixXd verts;
		Eigen::MatrixXi faces;
		MakeUnitCube(verts, faces);
		ConvertMatrix2SurfaceMesh(verts, faces, sm);
		auto oldCentroid = PMP::centroid(sm);
		auto normal = dir / sqrt(dir.squared_length());
		CGAL::Plane_3<Kernel> pl(center, normal);
		//auto oldCentroid = CGAL::centroid(sm.points().begin(), sm.points().end());
		for (auto vi : sm.vertices())
		{
			auto diff = sm.point(vi) - oldCentroid;
			sm.point(vi) = center + largeSideLen * diff.x() * pl.base1() + largeSideLen * diff.y() * pl.base2() + thickness * diff.z() * normal;
		}
		////////{
		////////	Surface_mesh input;
		////////	CGAL::IO::read_polygon_mesh("sphere.obj", input);
		////////	MakeLargePlate(0.1, inexact_Kernel::Point_3(0, 0, 0), inexact_Kernel::Vector_3(1, 1, 1), 100, sm);
		////////	PMP::corefine_and_compute_difference(input, sm, sm_out);
		////////	CGAL::IO::write_polygon_mesh("res.obj", sm_out);
		////////}
	}


	void Merge(const Eigen::MatrixXd& verts_in1, const Eigen::MatrixXi& faces_in1,
		const Eigen::MatrixXd& verts_in2, const Eigen::MatrixXi& faces_in2,
		Eigen::MatrixXd& verts_out, Eigen::MatrixXi& faces_out)
	{
		verts_out.resize(verts_in1.rows() + verts_in2.rows(), 3);
		verts_out.block(0, 0, verts_in1.rows(), 3) = verts_in1;
		verts_out.block(verts_in1.rows(), 0, verts_in2.rows(), 3) = verts_in2;
		faces_out.resize(faces_in1.rows() + faces_in2.rows(), 3);
		faces_out.block(0, 0, faces_in1.rows(), 3) = faces_in1;
		faces_out.block(faces_in1.rows(), 0, faces_in2.rows(), 3) = faces_in2.array() + verts_in1.rows();

		////////{
		////////	//����Merge��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	typedef CGAL::Polygon_2<double_Kernel> Polygon_2;
		////////	typedef CGAL::Polyhedron_3<double_Kernel> Polyhedron_3;
		////////	using namespace Model3D;
		////////	Eigen::MatrixXd V1;
		////////	Eigen::MatrixXi F1;
		////////	igl::readOBJ("sphere.obj", V1, F1);
		////////	Eigen::MatrixXd V2;
		////////	Eigen::MatrixXi F2;
		////////	igl::readOBJ("cube.obj", V2, F2);
		////////	V2 = (V2 * 0.5).array() + 0.9;
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	Merge(V1, F1, V2, F2, V, F);
		////////	igl::writeOBJ("merged.obj", V, F);
		////////}
	}

	void Merge(const list<pair<Eigen::MatrixXd, Eigen::MatrixXi>>& inputModels,
		Eigen::MatrixXd& verts_out, Eigen::MatrixXi& faces_out)
	{
		verts_out.resize(0, 3);
		faces_out.resize(0, 3);
		if (inputModels.empty())
			return;
		list<pair<Eigen::MatrixXd, Eigen::MatrixXi>>::const_iterator it = inputModels.begin();
		verts_out = it->first;
		faces_out = it->second;
		++it;
		while (it != inputModels.end())
		{
			auto verts_tmp = verts_out;
			auto faces_tmp = faces_out;
			Merge(verts_out, faces_out, it->first, it->second, verts_tmp, faces_tmp);
			swap(verts_tmp, verts_out);
			swap(faces_tmp, faces_out);
			++it;
		}
	}

	void RemoveUnreferencedVertices(const Eigen::MatrixXd& verts_in, const Eigen::MatrixXi& faces_in,
		Eigen::MatrixXd& verts_out, Eigen::MatrixXi& faces_out,
		vector<int>& mapNewVert2Old)
	{
		mapNewVert2Old.clear();
		map<int, int> fromOld2New;
		faces_out = faces_in;
		vector< Eigen::Vector3d> verts_new;
		for (int i = 0; i < faces_in.rows(); ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				if (fromOld2New.find(faces_in(i, j)) == fromOld2New.end())
				{
					verts_new.emplace_back(verts_in.row(faces_in(i, j)));
					fromOld2New[faces_in(i, j)] = verts_new.size() - 1;
					mapNewVert2Old.emplace_back(faces_in(i, j));
				}
				faces_out(i, j) = fromOld2New[faces_in(i, j)];
			}
		}
		verts_out.resize(verts_new.size(), 3);
		for (int i = 0; i < verts_new.size(); ++i)
			verts_out.row(i).array() = verts_new[i].array();

		////////{
		////////	//����RemoveUnreferencedVertices��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	typedef CGAL::Polygon_2<double_Kernel> Polygon_2;
		////////	typedef CGAL::Polyhedron_3<double_Kernel> Polyhedron_3;
		////////	using namespace Model3D;
		////////	Eigen::MatrixXd V1;
		////////	Eigen::MatrixXi F1;
		////////	igl::readOBJ("sphere.obj", V1, F1);
		////////	V1.conservativeResize(V1.rows() + 1, V1.cols());
		////////	V1.row(V1.rows() - 1).array() = Eigen::Vector3d(2, 2, 2);
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	RemoveUnreferencedVertices(V1, F1, V, F);
		////////	igl::writeOBJ("cleaned.obj", V, F);
		////////}
	}

	void RemoveUnreferencedVertices(const Eigen::MatrixXd& verts_in, const Eigen::MatrixXi& faces_in,
		Eigen::MatrixXd& verts_out, Eigen::MatrixXi& faces_out)
	{
		map<int, int> fromOld2New;
		faces_out = faces_in;
		vector< Eigen::Vector3d> verts_new;
		for (int i = 0; i < faces_in.rows(); ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				if (fromOld2New.find(faces_in(i, j)) == fromOld2New.end())
				{
					verts_new.emplace_back(verts_in.row(faces_in(i, j)));
					fromOld2New[faces_in(i, j)] = verts_new.size() - 1;
				}
				faces_out(i, j) = fromOld2New[faces_in(i, j)];
			}
		}
		verts_out.resize(verts_new.size(), 3);
		for (int i = 0; i < verts_new.size(); ++i)
			verts_out.row(i).array() = verts_new[i].array();

		////////{
		////////	//����RemoveUnreferencedVertices��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	typedef CGAL::Polygon_2<double_Kernel> Polygon_2;
		////////	typedef CGAL::Polyhedron_3<double_Kernel> Polyhedron_3;
		////////	using namespace Model3D;
		////////	Eigen::MatrixXd V1;
		////////	Eigen::MatrixXi F1;
		////////	igl::readOBJ("sphere.obj", V1, F1);
		////////	V1.conservativeResize(V1.rows() + 1, V1.cols());
		////////	V1.row(V1.rows() - 1).array() = Eigen::Vector3d(2, 2, 2);
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	RemoveUnreferencedVertices(V1, F1, V, F);
		////////	igl::writeOBJ("cleaned.obj", V, F);
		////////}
	}

	void ReverseOrientation(Eigen::MatrixXi& faces_in)
	{
		auto tmp = faces_in;
		faces_in.col(1) = tmp.col(2);
		faces_in.col(2) = tmp.col(1);

		//////////{
		//////////	//����ReverseOrientation��
		//////////	//non-manifold vertices, edges
		//////////	namespace PMP = CGAL::Polygon_mesh_processing;
		//////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		//////////	typedef CGAL::Polygon_2<double_Kernel> Polygon_2;
		//////////	typedef CGAL::Polyhedron_3<double_Kernel> Polyhedron_3;
		//////////	using namespace Model3D;

		//////////	Eigen::MatrixXd V;
		//////////	Eigen::MatrixXi F;
		//////////	igl::readOBJ("cube.obj", V, F);
		//////////	swap(F(0, 0), F(0, 1));
		//////////	igl::writeOBJ("cube2.obj", V, F);
		//////////	ForceConsistentOrientation(F);
		//////////	igl::writeOBJ("cube3.obj", V, F);
		//////////	ForceConsistentOrientation(V, F);
		//////////	igl::writeOBJ("cube4.obj", V, F);
		//////////}
	}

	void ForceConsistentOrientation(Eigen::MatrixXi& faces_in)
	{
		auto tmp = faces_in;
		Eigen::VectorXi C;
		igl::bfs_orient(tmp, faces_in, C);

		//////////{
		//////////	//����ForceConsistentOrientation��
		//////////	namespace PMP = CGAL::Polygon_mesh_processing;
		//////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		//////////	typedef CGAL::Polygon_2<double_Kernel> Polygon_2;
		//////////	typedef CGAL::Polyhedron_3<double_Kernel> Polyhedron_3;
		//////////	using namespace Model3D;

		//////////	Eigen::MatrixXd V;
		//////////	Eigen::MatrixXi F;
		//////////	igl::readOBJ("cube.obj", V, F);
		//////////	swap(F(0, 0), F(0, 1));
		//////////	igl::writeOBJ("cube2.obj", V, F);
		//////////	ForceConsistentOrientation(F);
		//////////	igl::writeOBJ("cube3.obj", V, F);
		//////////	ForceConsistentOrientation(V, F);
		//////////	igl::writeOBJ("cube4.obj", V, F);
		//////////}
	}

	void ForceConsistentOrientation(const Eigen::MatrixXd& V, Eigen::MatrixXi& faces_in)
	{
		auto tmp = faces_in;
		Eigen::VectorXi C;
		igl::bfs_orient(tmp, faces_in, C);
		Eigen::Vector3d center;
		double vol;
		igl::centroid(V, faces_in, center, vol);
		if (vol < 0)
		{
			ReverseOrientation(faces_in);
		}

		////////////{
		////////////	//����ForceConsistentOrientation��
		////////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////////	typedef CGAL::Polygon_2<double_Kernel> Polygon_2;
		////////////	typedef CGAL::Polyhedron_3<double_Kernel> Polyhedron_3;
		////////////	using namespace Model3D;

		////////////	Eigen::MatrixXd V;
		////////////	Eigen::MatrixXi F;
		////////////	igl::readOBJ("cube.obj", V, F);
		////////////	swap(F(0, 0), F(0, 1));
		////////////	igl::writeOBJ("cube2.obj", V, F);
		////////////	ForceConsistentOrientation(F);
		////////////	igl::writeOBJ("cube3.obj", V, F);
		////////////	ForceConsistentOrientation(V, F);
		////////////	igl::writeOBJ("cube4.obj", V, F);
		////////////}
	}

	void RepairNonmanifoldVerticesAndEdges(const Eigen::MatrixXd& verts_in, const Eigen::MatrixXi& faces_in,
		Eigen::MatrixXd& verts_out, Eigen::MatrixXi& faces_out,
		int smallestSize)
	{
		verts_out.resize(0, 3);
		faces_out.resize(0, 3);
		UnionFind uf_faces(faces_in.rows());
		map<pair<int, int>, set<int>> edgeOnWhichFaces;
		vector<int> onHowManyBoundaryEdges(verts_in.rows(), 0);
		set<int> boundaryFaces;
		for (int i = 0; i < faces_in.rows(); ++i)
		{
			edgeOnWhichFaces[make_pair(min(faces_in(i, 0), faces_in(i, 1)),
				max(faces_in(i, 0), faces_in(i, 1)))].insert(i);
			edgeOnWhichFaces[make_pair(min(faces_in(i, 1), faces_in(i, 2)),
				max(faces_in(i, 1), faces_in(i, 2)))].insert(i);
			edgeOnWhichFaces[make_pair(min(faces_in(i, 2), faces_in(i, 0)),
				max(faces_in(i, 2), faces_in(i, 0)))].insert(i);
		}
		for (auto mypair : edgeOnWhichFaces)
		{
			if (mypair.second.size() == 2)
			{
				uf_faces.AddConnection(*mypair.second.begin(), *mypair.second.rbegin());
			}
			if (mypair.second.size() == 1)
			{
				boundaryFaces.insert(*mypair.second.begin());
				onHowManyBoundaryEdges[mypair.first.first]++;
				onHowManyBoundaryEdges[mypair.first.second]++;
			}
		}
		set<int> singularVertices;
		for (int i = 0; i < onHowManyBoundaryEdges.size(); ++i)
		{
			if (onHowManyBoundaryEdges[i] > 2)
				singularVertices.insert(i);
		}
		auto verts_in_2 = verts_in;
		double ratio = 0.95;
		for (auto item : uf_faces.GetAllClusters())
		{
			if (item.second.size() <= smallestSize)
				continue;
			vector<Eigen::Vector3i> faces_cluster;
			for (auto id : item.second)
			{
				if (singularVertices.find(faces_in(id, 0)) == singularVertices.end()
					&& singularVertices.find(faces_in(id, 1)) == singularVertices.end()
					&& singularVertices.find(faces_in(id, 2)) == singularVertices.end())
				{
					faces_cluster.emplace_back(faces_in.row(id));
				}
#if 0
				else
				{
					int id0 = faces_in(id, 0);
					auto center = (verts_in.row(faces_in(id, 0)) + verts_in.row(faces_in(id, 1)) + verts_in.row(faces_in(id, 2))) / 3;
					if (singularVertices.find(faces_in(id, 0)) != singularVertices.end())
					{
						int oldSize = verts_in_2.rows();
						verts_in_2.conservativeResize(oldSize + 1, 3);
						verts_in_2.row(oldSize).array() = ratio * verts_in.row(id0) + (1 - ratio) * center;
						id0 = oldSize;
					}
					int id1 = faces_in(id, 1);
					if (singularVertices.find(faces_in(id, 1)) != singularVertices.end())
					{
						int oldSize = verts_in_2.rows();
						verts_in_2.conservativeResize(oldSize + 1, 3);
						verts_in_2.row(oldSize).array() = ratio * verts_in.row(id1) + (1 - ratio) * center;
						id1 = oldSize;
					}
					int id2 = faces_in(id, 2);
					if (singularVertices.find(faces_in(id, 2)) != singularVertices.end())
					{
						int oldSize = verts_in_2.rows();
						verts_in_2.conservativeResize(oldSize + 1, 3);
						verts_in_2.row(oldSize).array() = ratio * verts_in.row(id2) + (1 - ratio) * center;
						id2 = oldSize;
			}
					faces_cluster.push_back(Eigen::Vector3i(id0, id1, id2));
		}
#endif
	}

			Eigen::MatrixXi faces_tmp(faces_cluster.size(), 3);
			for (int i = 0; i < faces_cluster.size(); ++i)
			{
				faces_tmp.row(i).array() = faces_cluster[i].array();
			}
			Eigen::MatrixXd verts_new;
			Eigen::MatrixXi faces_new;
			RemoveUnreferencedVertices(verts_in_2, faces_tmp,
				verts_new, faces_new);
			ForceConsistentOrientation(faces_new);
			Eigen::MatrixXd verts_merged_tmp;
			Eigen::MatrixXi faces_merged_tmp;
			Merge(verts_new, faces_new, verts_out, faces_out, verts_merged_tmp, faces_merged_tmp);
			swap(verts_merged_tmp, verts_out);
			swap(faces_merged_tmp, faces_out);
}
		////////{
		////////	//����RepairNonmanifoldVerticesAndEdges��
		////////	//non-manifold vertices, edges
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	typedef CGAL::Polygon_2<double_Kernel> Polygon_2;
		////////	typedef CGAL::Polyhedron_3<double_Kernel> Polyhedron_3;
		////////	using namespace Model3D;
		////////	Eigen::MatrixXd V1;
		////////	Eigen::MatrixXi F1;
		////////	V1.resize(5, 3);
		////////	F1.resize(2, 3);
		////////	V1.row(0).array() = Eigen::Vector3d(-1, 1, 0); // #0
		////////	V1.row(1).array() = Eigen::Vector3d(0, 0, 0); // #1
		////////	V1.row(2).array() = Eigen::Vector3d(1, 1, 0); // #2
		////////	V1.row(3).array() = Eigen::Vector3d(-1, -1, 0); // #3
		////////	V1.row(4).array() = Eigen::Vector3d(1, -1, 0); // #4
		////////	F1.row(0).array() = Eigen::Vector3i(0, 1, 2); // #0
		////////	F1.row(1).array() = Eigen::Vector3i(1, 3, 4); // #1
		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	RepairNonmanifoldVerticesAndEdges(V1, F1, V, F, 0);
		////////	igl::writeOBJ("manifold1.obj", V, F);
		////////	V1.resize(5, 3);
		////////	F1.resize(3, 3);
		////////	V1.row(0).array() = Eigen::Vector3d(1, 0, 0); // #0
		////////	V1.row(1).array() = Eigen::Vector3d(0, 1, 0); // #1
		////////	V1.row(2).array() = Eigen::Vector3d(-1, 0, 0); // #2
		////////	V1.row(3).array() = Eigen::Vector3d(0, -1, 0); // #3
		////////	V1.row(4).array() = Eigen::Vector3d(0, 0, 1); // #4
		////////	F1.row(0).array() = Eigen::Vector3i(0, 1, 2); // #0
		////////	F1.row(1).array() = Eigen::Vector3i(0, 2, 3); // #1
		////////	F1.row(2).array() = Eigen::Vector3i(0, 2, 4); // #2
		////////	RepairNonmanifoldVerticesAndEdges(V1, F1, V, F, 0);
		////////	igl::writeOBJ("manifold2.obj", V, F);
		////////}
	}

	Eigen::Vector3d GetPosInNewSystem(Eigen::Vector3d p,
		Eigen::Vector3d origin, Eigen::Vector3d X, Eigen::Vector3d Y, Eigen::Vector3d Z)
	{
		return Eigen::Vector3d((p - origin).dot(X),
			(p - origin).dot(Y),
			(p - origin).dot(Z));
	}

	Eigen::Vector3d GetPosSpannedBySystem(Eigen::Vector3d p,
		Eigen::Vector3d origin, Eigen::Vector3d X, Eigen::Vector3d Y, Eigen::Vector3d Z)
	{
		return origin + p.x() * X + p.y() * Y + p.z() * Z;
	}

	void Relocate(Eigen::MatrixXd &V,
		Eigen::Vector3d origin, Eigen::Vector3d normal)
	{
		typedef CGAL::Cartesian<double> double_Kernel;
		typedef double_Kernel::Plane_3 Plane_3;
		typedef double_Kernel::Point_3 Point_3;
		typedef double_Kernel::Vector_3 Vector_3;
		Plane_3 pl(Point_3(origin.x(), origin.y(), origin.z()), Vector_3(normal.x(), normal.y(), normal.z()));
		for (int i = 0; i < V.rows(); ++i)
		{
			Eigen::RowVector3d vec = V.row(i);
			auto newPos = Point_3(origin.x(), origin.y(), origin.z()) + vec.x()* pl.base1() + vec.y() * pl.base2() + vec.z() * Vector_3(normal.x(), normal.y(), normal.z());
			V(i, 0) = newPos.x();
			V(i, 1) = newPos.y();
			V(i, 2) = newPos.z();
		}
	}


	void RemoveShortEdges(const Eigen::MatrixXd& verts_in, const Eigen::MatrixXi& faces_in,
		Eigen::MatrixXd& verts_out, Eigen::MatrixXi& faces_out,
		double coe)//remove edges if the length < coe * average_edge_length
	{
		double length_sum(0);
		int cnt(0);
		for (int i = 0; i < faces_in.rows(); ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				int nxt = (j + 1) % 3;
				double len = (verts_in.row(faces_in(i, j)) - verts_in.row(faces_in(i, nxt))).norm();
				length_sum += len;
				cnt++;
			}
		}
		double average_len = length_sum / cnt;

		UnionFind uf_vertices(verts_in.rows());
		for (int i = 0; i < faces_in.rows(); ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				int nxt = (j + 1) % 3;
				double len = (verts_in.row(faces_in(i, j)) - verts_in.row(faces_in(i, nxt))).norm();
				if (len < coe * average_len)
				{
					uf_vertices.AddConnection(faces_in(i, j), faces_in(i, nxt));
				}
			}
		}
		uf_vertices.UpdateParents2Ancestors();
		vector<Eigen::Vector3i> new_faces;
		UnionFind uf_faces(faces_in.rows());
		for (int i = 0; i < faces_in.rows(); ++i)
		{
			set<int> ids;
			for (int j = 0; j < 3; ++j)
			{
				ids.insert(uf_vertices.FindAncestor(faces_in(i, j)));
			}
			if (ids.size() < 3)
				continue;
			new_faces.emplace_back(Eigen::Vector3i(uf_vertices.FindAncestor(faces_in(i, 0)),
				uf_vertices.FindAncestor(faces_in(i, 1)),
				uf_vertices.FindAncestor(faces_in(i, 2))));
		}

		auto verts_tmp = verts_in;
		for (auto mypair : uf_vertices.GetNontrivialClusters())
		{
			Eigen::Vector3d averagePos = Eigen::Vector3d::Zero();
			for (int i = 0; i < mypair.second.size(); ++i)
			{
				averagePos = averagePos + Eigen::Vector3d(verts_in.row(mypair.second[i]));
			}
			verts_tmp.row(mypair.first).array() = averagePos / mypair.second.size();
		}
		Eigen::MatrixXi faces_tmp(new_faces.size(), 3);
		for (int i = 0; i < new_faces.size(); ++i)
		{
			faces_tmp.row(i).array() = new_faces[i].array();
		}
		RemoveUnreferencedVertices(verts_tmp, faces_tmp, verts_out, faces_out);
	}

	template<typename Kernel>
	void FillSmallHoles(CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm,
		int max_num_hole_edges = INT_MAX)
	{
		typedef CGAL::Point_3<Kernel> Point_3;
		typedef CGAL::Surface_mesh<Point_3>                           Mesh;
		typedef boost::graph_traits<Mesh>::vertex_descriptor        vertex_descriptor;
		typedef boost::graph_traits<Mesh>::halfedge_descriptor      halfedge_descriptor;
		typedef boost::graph_traits<Mesh>::face_descriptor          face_descriptor;
		namespace PMP = CGAL::Polygon_mesh_processing;

		auto is_small_hole = [=](halfedge_descriptor h, Mesh& mesh,
			int max_num_hole_edges)->bool
		{
			int num_hole_edges = 0;
			for (halfedge_descriptor hc : CGAL::halfedges_around_face(h, mesh))
			{
				// const Point& p = mesh.point(target(hc, mesh));
				++num_hole_edges;
				// Exit early, to avoid unnecessary traversal of large holes
				if (num_hole_edges > max_num_hole_edges) return false;
			}
			return true;
		};

		unsigned int nb_holes = 0;
		std::vector<halfedge_descriptor> border_cycles;
		// collect one halfedge per boundary cycle
		PMP::extract_boundary_cycles(sm, std::back_inserter(border_cycles));
		std::vector<face_descriptor>  patch_facets;
		for (halfedge_descriptor h : border_cycles)
		{
			if (!is_small_hole(h, sm, max_num_hole_edges))
				continue;
			PMP::triangulate_hole(sm, h, std::back_inserter(patch_facets));
			++nb_holes;
		}
		for (int i = 0; i < patch_facets.size(); ++i)
		{
			auto f = patch_facets[i];
			int j = 0;
			CGAL::Vertex_around_face_circulator<CGAL::Surface_mesh<Point_3>>
				vcirc(sm.halfedge(f), sm), done(vcirc);
			auto id1 = *vcirc;
			vcirc++;
			auto id2 = *vcirc;
			vcirc++;
			auto id3 = *vcirc;
			sm.add_face(id1, id2, id3);
		}

		////////{
		////////	//����FillSmallHoles��
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef CGAL::Surface_mesh<double_Kernel::Point_3> Surface_mesh;
		////////	typedef CGAL::Polygon_2<double_Kernel> Polygon_2;
		////////	typedef CGAL::Polyhedron_3<double_Kernel> Polyhedron_3;
		////////	using namespace Model3D;

		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	igl::readOBJ("lower_with_hole.obj", V, F);
		////////	Surface_mesh sm;
		////////	ConvertMatrix2SurfaceMesh(V, F, sm);
		////////	FillSmallHoles(sm, 100);
		////////	CGAL::IO::write_polygon_mesh("lower_without_hole.obj", sm);
		////////}
	}
	template<typename Kernel>
	void Remesh(const CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm_in,
		double coe, //target_edge_length = average_edgelength * coe;
		unsigned int nb_iter, //e.g. nb_iter = 3
		CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm_out)
	{
		typedef CGAL::Point_3<Kernel> Point_3;
		typedef CGAL::Surface_mesh<Point_3>                        Mesh;
		typedef boost::graph_traits<Mesh>::halfedge_descriptor        halfedge_descriptor;
		typedef boost::graph_traits<Mesh>::edge_descriptor            edge_descriptor;
		namespace PMP = CGAL::Polygon_mesh_processing;
		struct halfedge2edge
		{
			halfedge2edge(const Mesh& m, std::vector<edge_descriptor>& edges)
				: m_mesh(m), m_edges(edges)
			{}
			void operator()(const halfedge_descriptor& h) const
			{
				m_edges.emplace_back(edge(h, m_mesh));
			}
			const Mesh& m_mesh;
			std::vector<edge_descriptor>& m_edges;
		};
		double edgelength(0);
		auto r = sm_in.edges();
		// The iterators can be accessed through the C++ range API
		auto vb = r.begin();
		auto ve = r.end();
		// or with boost::tie, as the CGAL range derives from std::pair
		int cnt(0);
		for (boost::tie(vb, ve) = sm_in.edges(); vb != ve; ++vb) {
			auto vec = sm_in.point(sm_in.vertex(*vb, 1)) - sm_in.point(sm_in.vertex(*vb, 0));
			edgelength += sqrt(vec.squared_length());
			++cnt;
		}
		edgelength /= cnt;
		double target_edge_length = edgelength * coe;
		std::vector<edge_descriptor> border;
		sm_out = sm_in;
		PMP::border_halfedges(CGAL::faces(sm_out), sm_out, boost::make_function_output_iterator(halfedge2edge(sm_out, border)));
		PMP::split_long_edges(border, target_edge_length, sm_out);
		PMP::isotropic_remeshing(faces(sm_out), target_edge_length, sm_out,
			CGAL::parameters::number_of_iterations(nb_iter).protect_constraints(true));

		////////{
		////////	//����Remesh��
		////////	//must use: ************Exact_predicates_inexact_constructions_kernel**************
		////////	namespace PMP = CGAL::Polygon_mesh_processing;
		////////	typedef inexact_Kernel K;
		////////	typedef CGAL::Surface_mesh<K::Point_3> Surface_mesh;
		////////	typedef CGAL::Polygon_2<K> Polygon_2;
		////////	typedef CGAL::Polyhedron_3<K> Polyhedron_3;
		////////	using namespace Model3D;

		////////	Eigen::MatrixXd V;
		////////	Eigen::MatrixXi F;
		////////	igl::readOBJ("lower_without_hole.obj", V, F);
		////////	Eigen::MatrixXd V2;
		////////	Eigen::MatrixXi F2;
		////////	RemoveUnreferencedVertices(V, F, V2, F2);
		////////	swap(V, V2);
		////////	swap(F, F2);
		////////	igl::writeOBJ("lower_without_hole2.obj", V, F);
		////////	CRichModel model("lower_without_hole2.obj");
		////////	model.LoadModel();
		////////	model.PrintInfo(cerr);
		////////	Surface_mesh sm;
		////////	ConvertMatrix2SurfaceMesh(V, F, sm);
		////////	Surface_mesh sm_out;
		////////	Remesh(sm, 1, 3, sm_out);
		////////	CGAL::IO::write_polygon_mesh("lower_refined.obj", sm_out);
		////////}
	}


	template<typename Kernel>
	void AlignCentroidTo(CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm, CGAL::Point_3<Kernel> newCentroid)
	{
		typedef CGAL::Point_3<Kernel> Point_3;
		namespace PMP = CGAL::Polygon_mesh_processing;
		auto oldCentroid = CGAL::centroid(sm.points().begin(), sm.points().end());
		for (auto vi : sm.vertices())
		{
			sm.point(vi) = newCentroid + (sm.point(vi) - oldCentroid);
		}
	}

	template<typename Kernel>
	void MakeUnitCube_HighQuality(CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm)
	{
		typedef CGAL::Point_3<Kernel> Point_3;
		namespace PMP = CGAL::Polygon_mesh_processing;
		CGAL::Surface_mesh<Point_3> sm_bottom, sm_top, sm_left, sm_right, sm_front, sm_back;
		MakeUnitSquare(sm);
		auto centroid = CGAL::centroid(sm.points().begin(), sm.points().end());
		for (auto vi : sm.vertices())
		{
			sm.point(vi) = centroid + (sm.point(vi) - centroid) * 1.2;
		}
		Remesh(sm, 0.2, 3, sm_bottom);
		PMP::reverse_face_orientations(sm_bottom);
		sm_top = sm_bottom;
		AlignCentroidTo(sm_top, Point_3(0.5, 0.5, 1));
		PMP::reverse_face_orientations(sm_top);


		auto AlternateSurfaceMesh = [](CGAL::Surface_mesh<Point_3>& sm)
		{
			auto AlternateXYZ = [](Point_3 p)->Point_3
			{
				return Point_3(p.z(), p.x(), p.y());
			};
			for (auto vi : sm.vertices())
			{
				sm.point(vi) = AlternateXYZ(sm.point(vi));
			}
		};
		sm_left = sm_bottom;
		AlternateSurfaceMesh(sm_left);
		sm_right = sm_top;
		AlternateSurfaceMesh(sm_right);
		sm_front = sm_left;
		AlternateSurfaceMesh(sm_front);
		sm_back = sm_right;
		AlternateSurfaceMesh(sm_back);

		{
			PMP::corefine(sm_left, sm_bottom);
			PMP::corefine(sm_left, sm_top);
			PMP::corefine(sm_left, sm_front);
			PMP::corefine(sm_left, sm_back);
			PMP::corefine(sm_right, sm_bottom);
			PMP::corefine(sm_right, sm_top);
			PMP::corefine(sm_right, sm_front);
			PMP::corefine(sm_right, sm_back);
			PMP::corefine(sm_front, sm_top);
			PMP::corefine(sm_back, sm_top);
			PMP::corefine(sm_front, sm_bottom);
			PMP::corefine(sm_back, sm_bottom);

			std::vector<Point_3 > points;
			std::vector<std::vector<int> > triangles;

			auto AddNecessaryTriangles = [&](const CGAL::Surface_mesh<Point_3>& sm)
			{
				for (auto face_index : sm.faces())
				{
					Point_3 pts[3];
					int j = 0;
					CGAL::Vertex_around_face_circulator<CGAL::Surface_mesh<Point_3>> vcirc(sm.halfedge(face_index), sm), done(vcirc);
					do
					{
						pts[j] = sm.point(*vcirc++);
						++j;
					} while (vcirc != done);
					auto centroid = CGAL::centroid(pts[0], pts[1], pts[2]);
					if (centroid.x() > -1e-7 && centroid.x() < 1 + 1e-7
						&& centroid.y() > -1e-7 && centroid.y() < 1 + 1e-7
						&& centroid.z() > -1e-7 && centroid.z() < 1 + 1e-7)
					{
						points.emplace_back(pts[0]);
						points.emplace_back(pts[1]);
						points.emplace_back(pts[2]);
						vector<int> ids;
						ids.emplace_back((int)points.size() - 3);
						ids.emplace_back((int)points.size() - 2);
						ids.emplace_back((int)points.size() - 1);
						triangles.emplace_back(ids);
					}
				}
			};
			AddNecessaryTriangles(sm_left);
			AddNecessaryTriangles(sm_right);
			AddNecessaryTriangles(sm_top);
			AddNecessaryTriangles(sm_bottom);
			AddNecessaryTriangles(sm_front);
			AddNecessaryTriangles(sm_back);
			PMP::repair_polygon_soup(points, triangles);
			sm.clear();
			PMP::polygon_soup_to_polygon_mesh(points, triangles, sm);
		}
		{
			////////////���Դ���
			////////////MakeUnitCube_HighQuality(sm_out);
			////////////CGAL::IO::write_polygon_mesh("res.obj", sm_out);
			////////////CRichModel model("res.obj");
			////////////model.LoadModel();
			////////////model.PrintInfo(cerr);
		}
	}

	//remove unreferenced vertices
	pair<vector<Model3D::CPoint3D>, vector<Model3D::CBaseModel::CFace>>
		RemoveUnreferencedVertices(const vector<Model3D::CPoint3D>& verts, const vector<Model3D::CBaseModel::CFace>& faces,
			vector<int>& mapNewVert2Old)
	{
		mapNewVert2Old.clear();
		map<int, int> fromOld2New;
		vector<Model3D::CPoint3D> vert_tmp;
		vector<Model3D::CBaseModel::CFace> faces_tmp;
		for (auto face : faces)
		{
			for (int i = 0; i < 3; ++i)
			{
				int cnt(fromOld2New.size());
				if (fromOld2New.find(face[i]) == fromOld2New.end())
				{
					fromOld2New[face[i]] = cnt;
					vert_tmp.emplace_back(verts[face[i]]);
					mapNewVert2Old.emplace_back(face[i]);
				}
			}
			faces_tmp.emplace_back(Model3D::CBaseModel::CFace(fromOld2New.find(face[0])->second,
				fromOld2New.find(face[1])->second,
				fromOld2New.find(face[2])->second));
		}
		return make_pair(vert_tmp, faces_tmp);
		//����ʡ��
	}
	//remove unreferenced vertices
	pair< vector<Model3D::CPoint3D>, vector<Model3D::CBaseModel::CFace>>
		RemoveUnreferencedVertices(const vector<Model3D::CPoint3D>& verts, const vector<Model3D::CBaseModel::CFace>& faces)
	{
		map<int, int> fromOld2New;
		vector<Model3D::CPoint3D> vert_tmp;
		vector<Model3D::CBaseModel::CFace> faces_tmp;
		for (auto face : faces)
		{
			for (int i = 0; i < 3; ++i)
			{
				int cnt(fromOld2New.size());
				if (fromOld2New.find(face[i]) == fromOld2New.end())
				{
					fromOld2New[face[i]] = cnt;
					vert_tmp.emplace_back(verts[face[i]]);
				}
			}
			faces_tmp.emplace_back(Model3D::CBaseModel::CFace(fromOld2New.find(face[0])->second,
				fromOld2New.find(face[1])->second,
				fromOld2New.find(face[2])->second));
		}
		return make_pair(vert_tmp, faces_tmp);
		//����ʡ��
	}

	pair< vector<Model3D::CPoint3D>, vector<Model3D::CBaseModel::CFace>>
		Merge(const list<pair< vector<Model3D::CPoint3D>, vector<Model3D::CBaseModel::CFace>>>& models)
	{
		pair< vector<Model3D::CPoint3D>, vector<Model3D::CBaseModel::CFace>> res;
		for (auto item : models)
		{
			int cnt(res.first.size());
			copy(item.first.begin(), item.first.end(), back_inserter(res.first));
			for (auto face : item.second)
			{
				res.second.emplace_back(Model3D::CBaseModel::CFace(face[0] + cnt,
					face[1] + cnt,
					face[2] + cnt));
			}
		}
		return res;
	}

	void ForceConsistentOrientation(vector<Model3D::CBaseModel::CFace>& faces)
	{
		Eigen::MatrixXi F(faces.size(), 3);
		for (int i = 0; i < faces.size(); ++i)
		{
			for (int j = 0; j < 3; ++j)
				F(i, j) = faces[i][j];
		}
		ForceConsistentOrientation(F);
		for (int i = 0; i < faces.size(); ++i)
		{
			for (int j = 0; j < 3; ++j)
				faces[i][j] = F(i, j);
		}
	}

	void ForceConsistentOrientation(const vector<Model3D::CPoint3D>& verts,
		vector<Model3D::CBaseModel::CFace>& faces)
	{
		Eigen::MatrixXd V(verts.size(), 3);
		for (int i = 0; i < verts.size(); ++i)
		{
			V(i, 0) = verts[i].x;
			V(i, 1) = verts[i].y;
			V(i, 2) = verts[i].z;
		}
		Eigen::MatrixXi F(faces.size(), 3);
		for (int i = 0; i < faces.size(); ++i)
		{
			for (int j = 0; j < 3; ++j)
				F(i, j) = faces[i][j];
		}
		ForceConsistentOrientation(V, F);
		for (int i = 0; i < faces.size(); ++i)
		{
			for (int j = 0; j < 3; ++j)
				faces[i][j] = F(i, j);
		}
	}

	list<pair< vector<Model3D::CPoint3D>, vector<Model3D::CBaseModel::CFace>>>
		RepairNonmanifoldVerticesAndEdges(const vector<Model3D::CPoint3D>& verts, const vector<Model3D::CBaseModel::CFace>& faces, int smallestSize)
	{
		UnionFind uf_faces(faces.size());
		map<pair<int, int>, set<int>> edgeOnWhichFaces;
		vector<int> onHowManyBoundaryEdges(verts.size(), 0);
		set<int> boundaryFaces;
		for (int i = 0; i < faces.size(); ++i)
		{
			edgeOnWhichFaces[make_pair(min(faces[i][0], faces[i][1]),
				max(faces[i][0], faces[i][1]))].insert(i);
			edgeOnWhichFaces[make_pair(min(faces[i][1], faces[i][2]),
				max(faces[i][1], faces[i][2]))].insert(i);
			edgeOnWhichFaces[make_pair(min(faces[i][2], faces[i][0]),
				max(faces[i][2], faces[i][0]))].insert(i);
		}
		for (auto mypair : edgeOnWhichFaces)
		{
			if (mypair.second.size() == 2)
			{
				uf_faces.AddConnection(*mypair.second.begin(), *mypair.second.rbegin());
			}
			if (mypair.second.size() == 1)
			{
				boundaryFaces.insert(*mypair.second.begin());
				onHowManyBoundaryEdges[mypair.first.first]++;
				onHowManyBoundaryEdges[mypair.first.second]++;
			}
		}
		set<int> singularVertices;
		for (int i = 0; i < onHowManyBoundaryEdges.size(); ++i)
		{
			if (onHowManyBoundaryEdges[i] > 2)
				singularVertices.insert(i);
		}

		//vector<CBaseModel::CFace> faces_augumented;
		//for (int i = 0; i < faces.size(); ++i)
		//{
		//	if (singularVertices.find(faces[i][0]) == singularVertices.end()
		//		&& singularVertices.find(faces[i][1]) == singularVertices.end()
		//		&& singularVertices.find(faces[i][2]) == singularVertices.end())
		//	{
		//		faces_augumented.push_back(faces[i]);
		//	}
		//}

		auto verts_2 = verts;
		double ratio = 0.95;
		list<pair< vector<Model3D::CPoint3D>, vector<Model3D::CBaseModel::CFace>>> res;
		for (auto item : uf_faces.GetNontrivialClusters())
		{
			if (item.second.size() < smallestSize)
				continue;
			vector<Model3D::CBaseModel::CFace> faces_cluster;
			for (auto id : item.second)
			{
				if (singularVertices.find(faces[id][0]) == singularVertices.end()
					&& singularVertices.find(faces[id][1]) == singularVertices.end()
					&& singularVertices.find(faces[id][2]) == singularVertices.end())
				{
					faces_cluster.emplace_back(faces[id]);
				}
				//else
				//{
				//	int id0 = faces[id][0];
				//	auto center = (verts[faces[id][0]] + verts[faces[id][1]] + verts[faces[id][2]]) / 3;
				//	if (singularVertices.find(faces[id][0]) != singularVertices.end())
				//	{
				//		int oldSize = verts_2.size();
				//		verts_2.push_back(ratio * verts[faces[id][0]] + (1 - ratio) * center);
				//		id0 = oldSize;
				//	}
				//	int id1 = faces[id][1];
				//	if (singularVertices.find(faces[id][1]) != singularVertices.end())
				//	{
				//		int oldSize = verts_2.size();
				//		verts_2.push_back(ratio * verts[faces[id][1]] + (1 - ratio) * center);
				//		id1 = oldSize;
				//	}
				//	int id2 = faces[id][2];
				//	if (singularVertices.find(faces[id][2]) != singularVertices.end())
				//	{
				//		int oldSize = verts_2.size();
				//		verts_2.push_back(ratio * verts[faces[id][2]] + (1 - ratio) * center);
				//		id1 = oldSize;
				//	}
				//	faces_cluster.push_back(Model3D::CBaseModel::CFace(id0, id1, id2));
				//}
			}
			auto component = RemoveUnreferencedVertices(verts_2, faces_cluster);
			ForceConsistentOrientation(component.first, component.second);
			res.emplace_back(component);
		}
		return res;
	}

	pair< vector<Model3D::CPoint3D>, vector<Model3D::CBaseModel::CFace>> RemoveShortEdges(const vector<Model3D::CPoint3D>& verts,
		const vector<Model3D::CBaseModel::CFace>& faces, double coe)
	{
		double length_sum(0);
		int cnt(0);
		for (int i = 0; i < faces.size(); ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				int nxt = (j + 1) % 3;
				double len = (verts[faces[i][j]] - verts[faces[i][nxt]]).Len();
				length_sum += len;
				cnt++;
			}
		}
		double average_len = length_sum / cnt;
		UnionFind uf_vertices(verts.size());
		for (int i = 0; i < faces.size(); ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				int nxt = (j + 1) % 3;
				double len = (verts[faces[i][j]] - verts[faces[i][nxt]]).Len();
				if (len < coe * average_len)
				{
					uf_vertices.AddConnection(faces[i][j], faces[i][nxt]);
				}
			}
		}
		uf_vertices.UpdateParents2Ancestors();

		vector<Model3D::CBaseModel::CFace> new_faces;
		UnionFind uf_faces(faces.size());


		for (int i = 0; i < faces.size(); ++i)
		{
			set<int> ids;
			for (int j = 0; j < 3; ++j)
			{
				ids.insert(uf_vertices.FindAncestor(faces[i][j]));
			}
			if (ids.size() < 3)
				continue;
			new_faces.emplace_back(Model3D::CBaseModel::CFace(uf_vertices.FindAncestor(faces[i][0]),
				uf_vertices.FindAncestor(faces[i][1]), uf_vertices.FindAncestor(faces[i][2])));
		}

		vector<Model3D::CPoint3D> verts_tmp(verts);
		for (auto mypair : uf_vertices.GetNontrivialClusters())
		{
			Model3D::CPoint3D ptSum(Model3D::CPoint3D::Origin());
			for (int i = 0; i < mypair.second.size(); ++i)
			{
				ptSum = ptSum + verts[mypair.second[i]];
			}
			verts_tmp[mypair.first] = ptSum / mypair.second.size();
		}

		return RemoveUnreferencedVertices(verts_tmp, new_faces);
	}

	pair<Model3D::CBaseModel, Model3D::CBaseModel> SplitBasedOnScalarField_into_Two_Models(const Model3D::CBaseModel& model, const vector<double>& scalarField,
		double val)
	{
		double maxV = *max_element(scalarField.begin(), scalarField.end());
		double minV = *min_element(scalarField.begin(), scalarField.end());
		if (maxV < val)
		{
			return make_pair(model, Model3D::CBaseModel());
		}
		if (minV > val)
		{
			return make_pair(Model3D::CBaseModel(), model);
		}
		//model.SaveObjFile("observe.obj");
		Model3D::CRichModel clone(model);
		vector<Model3D::EdgePoint> eps;
		for (int i = 0; i < clone.GetNumOfEdges(); ++i)
		{
			if (clone.Edge(i).indexOfLeftVert >
				clone.Edge(i).indexOfRightVert)
				continue;
			if (scalarField[clone.Edge(i).indexOfLeftVert] >= val
				== scalarField[clone.Edge(i).indexOfRightVert] >= val)
				continue;
			double prop = (val - scalarField[clone.Edge(i).indexOfLeftVert])
				/ (scalarField[clone.Edge(i).indexOfRightVert] - scalarField[clone.Edge(i).indexOfLeftVert]);
			if (prop < 1e-5 || prop > 1 - 1e-5)
				continue;
			eps.emplace_back(Model3D::EdgePoint(i, prop));
		}
		int oldVertNum = model.GetNumOfVerts();
		for (int i = 0; i < eps.size(); ++i)
			clone.SplitEdge(eps[i]);
		vector<Model3D::CPoint3D> vertList(clone.m_Verts);
		vector<Model3D::CBaseModel::CFace> faceList_small;
		vector<Model3D::CBaseModel::CFace> faceList_large;
		for (int i = 0; i < clone.GetNumOfFaces(); ++i)
		{
			if (clone.m_UselessFaces.find(i) != clone.m_UselessFaces.end())
				continue;
			int v1 = clone.Face(i)[0];
			int v2 = clone.Face(i)[1];
			int v3 = clone.Face(i)[2];
			int cnt(0);
			double average(0);
			if (v1 < oldVertNum)
			{
				average += scalarField[v1];
				cnt++;
			}
			if (v2 < oldVertNum)
			{
				average += scalarField[v2];
				cnt++;
			}
			if (v3 < oldVertNum)
			{
				average += scalarField[v3];
				cnt++;
			}
			average /= cnt;
			if (average < val)
			{
				faceList_small.emplace_back(clone.Face(i));
			}
			else
			{
				faceList_large.emplace_back(clone.Face(i));
			}
		}

		//auto small = RemoveShortEdges(vertList, faceList_small, 0.02);
		//small = Merge(RepairNonmanifoldVerticesAndEdges(small.first, small.second, 4));
		//auto large = RemoveShortEdges(vertList, faceList_large, 0.02);
		//large = Merge(RepairNonmanifoldVerticesAndEdges(large.first, large.second, 4));
		//return make_pair(Model3D::CBaseModel(small.first, small.second),
		//	Model3D::CBaseModel(large.first, large.second));
		return make_pair(Model3D::CBaseModel(vertList, faceList_small),
			Model3D::CBaseModel(vertList, faceList_large));
	}

	pair< vector<Model3D::CPoint3D>, vector<Model3D::CBaseModel::CFace>>
		FillPlanarHolesWithoutAddingPoints(const vector<Model3D::CPoint3D>& verts,
			const vector<Model3D::CBaseModel::CFace>& faces, int largestSize = INT_MAX)
	{
		//for filling those holes that are nearly planar....
		map<pair<int, int>, set<int>> edgeOnWhichFaces;

		for (int i = 0; i < faces.size(); ++i)
		{
			edgeOnWhichFaces[make_pair(min(faces[i][0], faces[i][1]),
				max(faces[i][0], faces[i][1]))].insert(i);
			edgeOnWhichFaces[make_pair(min(faces[i][1], faces[i][2]),
				max(faces[i][1], faces[i][2]))].insert(i);
			edgeOnWhichFaces[make_pair(min(faces[i][2], faces[i][0]),
				max(faces[i][2], faces[i][0]))].insert(i);
		}

		UnionFind uf(verts.size());
		map<int, set<int>>neighs;
		set<int> boundaryVerts;
		for (auto item : edgeOnWhichFaces)
		{
			if (item.second.size() == 1)
			{
				uf.AddConnection(item.first.first, item.first.second);
				neighs[item.first.first].insert(item.first.second);
				neighs[item.first.second].insert(item.first.first);
				boundaryVerts.insert(item.first.first);
				boundaryVerts.insert(item.first.second);
			}
		}
		map<int, int> nextV;
		for (int i = 0; i < faces.size(); ++i)
		{
			if (boundaryVerts.find(faces[i][0]) != boundaryVerts.end()
				&& boundaryVerts.find(faces[i][1]) != boundaryVerts.end()
				&& edgeOnWhichFaces[make_pair(min(faces[i][0], faces[i][1]),
					max(faces[i][0], faces[i][1]))].size() == 1)
			{
				nextV[faces[i][1]] = faces[i][0];
			}
			if (boundaryVerts.find(faces[i][1]) != boundaryVerts.end()
				&& boundaryVerts.find(faces[i][2]) != boundaryVerts.end()
				&& edgeOnWhichFaces[make_pair(min(faces[i][1], faces[i][2]),
					max(faces[i][1], faces[i][2]))].size() == 1)
			{
				nextV[faces[i][2]] = faces[i][1];
			}
			if (boundaryVerts.find(faces[i][2]) != boundaryVerts.end()
				&& boundaryVerts.find(faces[i][0]) != boundaryVerts.end()
				&& edgeOnWhichFaces[make_pair(min(faces[i][0], faces[i][2]),
					max(faces[i][0], faces[i][2]))].size() == 1)
			{
				nextV[faces[i][0]] = faces[i][2];
			}
		}
		vector<Model3D::CBaseModel::CFace> newFaces(faces);
		for (auto item : uf.GetNontrivialClusters())
		{
			int key = item.first;
			if (item.second.size() > largestSize)
				continue;
			//for (int k = 0; k < 20; ++k)
			//	key = nextV[key];
			int v = key;
			vector< int> pt3d_list;
			//cerr << "------------" << endl;
			do
			{
				pt3d_list.emplace_back(v);
				//cerr << "v = " << v << endl;
				int next = nextV[v];
				neighs[next].erase(v);
				neighs[v].erase(next);
				v = next;
			} while (v != key);

			Model3D::CPoint3D vecSum(Model3D::CPoint3D::Origin());
			for (int i = 2; i < pt3d_list.size(); ++i)
			{
				vecSum = vecSum + VectorCross(verts[pt3d_list[0]],
					verts[pt3d_list[i - 1]],
					verts[pt3d_list[i]]);
			}
			vecSum.Normalize();
			Model3D::CPoint3D dir1 = vecSum.GetUnitPerpendicularDir();
			Model3D::CPoint3D dir2 = vecSum * dir1;
			vector<CDT_Kernel::Point_2> pt2d_list;
			map<CDT_Kernel::Point_2, int> fromPoint2ID;
			for (int i = 0; i < pt3d_list.size(); ++i)
			{
				pt2d_list.emplace_back(CDT_Kernel::Point_2(verts[pt3d_list[i]] ^ dir1, verts[pt3d_list[i]] ^ dir2));
				fromPoint2ID[pt2d_list.back()] = pt3d_list[i];
			}

			CDT cdt;
			cdt.insert_constraint(pt2d_list.begin(), pt2d_list.end(), true);
			mark_domains(cdt);
			for (CDT::Finite_faces_iterator fit = cdt.finite_faces_begin();
				fit != cdt.finite_faces_end(); ++fit)
			{
				if (fit->info().in_domain())
				{
					int id1 = fromPoint2ID[fit->vertex(0)->point()];
					int id2 = fromPoint2ID[fit->vertex(1)->point()];
					int id3 = fromPoint2ID[fit->vertex(2)->point()];
					//cerr << "ids : " << id1 << ", " << id2 << ", " << id3 << endl;					
					newFaces.emplace_back(Model3D::CBaseModel::CFace(id1, id2, id3));
				}
			}
		}
		return Merge(RepairNonmanifoldVerticesAndEdges(verts, newFaces, 3));

		//////////{
		//////////	//���ԣ�FillPlanarHolesWithoutAddingPoints
		//////////	namespace PMP = CGAL::Polygon_mesh_processing;
		//////////	typedef inexact_Kernel K;
		//////////	typedef CGAL::Surface_mesh<K::Point_3> Surface_mesh;
		//////////	typedef CGAL::Polygon_2<K> Polygon_2;
		//////////	typedef CGAL::Polyhedron_3<K> Polyhedron_3;
		//////////	using namespace Model3D;
		//////////	typedef K::Point_2                   Point;
		//////////	typedef CGAL::Straight_skeleton_2<K> Ss;
		//////////	typedef boost::shared_ptr<Ss> SsPtr;
		//////////	Polygon_2 poly;
		//////////	poly.push_back(Point(-1, -1));
		//////////	poly.push_back(Point(0, -12));
		//////////	poly.push_back(Point(1, -1));
		//////////	poly.push_back(Point(12, 0));
		//////////	poly.push_back(Point(1, 1));
		//////////	poly.push_back(Point(0, 12));
		//////////	poly.push_back(Point(-1, 1));
		//////////	poly.push_back(Point(-12, 0));
		//////////	CRichModel model("sphere_half.obj");
		//////////	model.LoadModel();
		//////////	auto res = FillPlanarHolesWithoutAddingPoints(model.m_Verts, model.m_Faces, model.GetNumOfVerts());
		//////////	CBaseModel(res.first, res.second).SaveObjFile("sphere_out.obj");
		//////////}
	}

	pair< vector<Model3D::CPoint3D>, vector<Model3D::CBaseModel::CFace>>
		FillPlanarHolesWithAddingSkeletalPoints(const vector<Model3D::CPoint3D>& verts,
			const vector<Model3D::CBaseModel::CFace>& faces, int largestSize = INT_MAX)
	{
		//for filling those holes that are nearly planar....
		map<pair<int, int>, set<int>> edgeOnWhichFaces;

		for (int i = 0; i < faces.size(); ++i)
		{
			edgeOnWhichFaces[make_pair(min(faces[i][0], faces[i][1]),
				max(faces[i][0], faces[i][1]))].insert(i);
			edgeOnWhichFaces[make_pair(min(faces[i][1], faces[i][2]),
				max(faces[i][1], faces[i][2]))].insert(i);
			edgeOnWhichFaces[make_pair(min(faces[i][2], faces[i][0]),
				max(faces[i][2], faces[i][0]))].insert(i);
		}

		UnionFind uf(verts.size());
		map<int, set<int>>neighs;
		set<int> boundaryVerts;
		for (auto item : edgeOnWhichFaces)
		{
			if (item.second.size() == 1)
			{
				uf.AddConnection(item.first.first, item.first.second);
				neighs[item.first.first].insert(item.first.second);
				neighs[item.first.second].insert(item.first.first);
				boundaryVerts.insert(item.first.first);
				boundaryVerts.insert(item.first.second);
			}
		}
		map<int, int> nextV;
		for (int i = 0; i < faces.size(); ++i)
		{
			if (boundaryVerts.find(faces[i][0]) != boundaryVerts.end()
				&& boundaryVerts.find(faces[i][1]) != boundaryVerts.end()
				&& edgeOnWhichFaces[make_pair(min(faces[i][0], faces[i][1]),
					max(faces[i][0], faces[i][1]))].size() == 1)
			{
				nextV[faces[i][1]] = faces[i][0];
			}
			if (boundaryVerts.find(faces[i][1]) != boundaryVerts.end()
				&& boundaryVerts.find(faces[i][2]) != boundaryVerts.end()
				&& edgeOnWhichFaces[make_pair(min(faces[i][1], faces[i][2]),
					max(faces[i][1], faces[i][2]))].size() == 1)
			{
				nextV[faces[i][2]] = faces[i][1];
			}
			if (boundaryVerts.find(faces[i][2]) != boundaryVerts.end()
				&& boundaryVerts.find(faces[i][0]) != boundaryVerts.end()
				&& edgeOnWhichFaces[make_pair(min(faces[i][0], faces[i][2]),
					max(faces[i][0], faces[i][2]))].size() == 1)
			{
				nextV[faces[i][0]] = faces[i][2];
			}
		}
		auto verts_cpy = verts;
		vector<Model3D::CBaseModel::CFace> newFaces(faces);
		for (auto item : uf.GetNontrivialClusters())
		{
			int key = item.first;
			if (item.second.size() > largestSize)
				continue;
			//for (int k = 0; k < 20; ++k)
			//	key = nextV[key];
			int v = key;
			vector< int> pt3d_list;
			//cerr << "------------" << endl;
			do
			{
				pt3d_list.emplace_back(v);
				//cerr << "v = " << v << endl;
				int next = nextV[v];
				neighs[next].erase(v);
				neighs[v].erase(next);
				v = next;
			} while (v != key);

			Model3D::CPoint3D vecSum(Model3D::CPoint3D::Origin());
			for (int i = 2; i < pt3d_list.size(); ++i)
			{
				vecSum = vecSum + VectorCross(verts[pt3d_list[0]],
					verts[pt3d_list[i - 1]],
					verts[pt3d_list[i]]);
			}
			vecSum.Normalize();
			Model3D::CPoint3D dir1 = vecSum.GetUnitPerpendicularDir();
			Model3D::CPoint3D dir2 = vecSum * dir1;
			vector<CDT_Kernel::Point_2> pt2d_list;
			map<CDT_Kernel::Point_2, int> fromPoint2ID;
			for (int i = 0; i < pt3d_list.size(); ++i)
			{
				pt2d_list.emplace_back(CDT_Kernel::Point_2((verts[pt3d_list[i]] - verts[pt3d_list[0]]) ^ dir1,
					(verts[pt3d_list[i]] - verts[pt3d_list[0]]) ^ dir2));
				fromPoint2ID[pt2d_list.back()] = pt3d_list[i];
			}

			CDT cdt;
			cdt.insert_constraint(pt2d_list.begin(), pt2d_list.end(), true);
			typedef CGAL::Straight_skeleton_2<CDT_Kernel> Ss;
			typedef boost::shared_ptr<Ss> SsPtr;
			SsPtr iss = CGAL::create_interior_straight_skeleton_2(pt2d_list.begin(), pt2d_list.end());
			//set<CDT_Kernel::Point_2> candidates;
			//for (auto q = iss->vertices_begin(); q != iss->vertices_end(); ++q)
			//{
			//	if (q->is_skeleton())
			//	{
			//		candidates.insert(q->point());
			//	}
			//}

			int cnt(0);
			double ave_len(0);
			for (auto e = iss->halfedges_begin(); e != iss->halfedges_end(); ++e)
			{
				if (e->is_border())
				{
					auto p_start = e->vertex()->point();
					auto p_end = e->opposite()->vertex()->point();
					ave_len += sqrt((p_start - p_end).squared_length());
					++cnt;
				}
			}
			ave_len /= cnt;

			set<CDT_Kernel::Point_2> candidates;
			for (auto e = iss->halfedges_begin(); e != iss->halfedges_end(); ++e)
			{
				auto p_start = e->vertex()->point();
				auto p_end = e->opposite()->vertex()->point();
				double len = sqrt((p_start - p_end).squared_length());
				if (p_start > p_end)
					continue;
				if (e->is_bisector())
				{
					int segNum = int(len / (2 * ave_len)) + 1;
					for (int j = 1; j < segNum; ++j)
					{
						candidates.insert(p_start + j / (double)segNum * (p_end - p_start));
					}
					if (e->vertex()->is_skeleton())
						candidates.insert(e->vertex()->point());
					if (e->opposite()->vertex()->is_skeleton())
						candidates.insert(e->opposite()->vertex()->point());
				}
			}
			set<CDT_Kernel::Point_2>  skeletalPoints;
			for (auto p : candidates)
			{
				bool flag(true);
				for (auto q : skeletalPoints)
				{
					if (sqrt((p - q).squared_length()) < ave_len)
					{
						flag = false;
						break;
					}
				}
				if (flag)
					skeletalPoints.insert(p);
			}
			for (auto q : skeletalPoints)
			{
				cdt.insert(q);
				Model3D::CPoint3D q3d = verts[pt3d_list[0]]
					+ q.x() * dir1
					+ q.y() * dir2;
				verts_cpy.emplace_back(q3d);
				fromPoint2ID[q] = verts_cpy.size() - 1;
			}

			mark_domains(cdt);
			for (CDT::Finite_faces_iterator fit = cdt.finite_faces_begin();
				fit != cdt.finite_faces_end(); ++fit)
			{
				if (fit->info().in_domain())
				{
					int id1 = fromPoint2ID[fit->vertex(0)->point()];
					int id2 = fromPoint2ID[fit->vertex(1)->point()];
					int id3 = fromPoint2ID[fit->vertex(2)->point()];
					//cerr << "ids : " << id1 << ", " << id2 << ", " << id3 << endl;					
					newFaces.emplace_back(Model3D::CBaseModel::CFace(id1, id2, id3));
				}
			}
		}
		return Merge(RepairNonmanifoldVerticesAndEdges(verts_cpy, newFaces, 3));
		////////	{
		////////		//���ԣ�FillPlanarHolesWithAddingSkeletalPoints
		////////		namespace PMP = CGAL::Polygon_mesh_processing;
		////////		typedef inexact_Kernel K;
		////////		typedef CGAL::Surface_mesh<K::Point_3> Surface_mesh;
		////////		typedef CGAL::Polygon_2<K> Polygon_2;
		////////		typedef CGAL::Polyhedron_3<K> Polyhedron_3;
		////////		using namespace Model3D;
		////////		typedef K::Point_2                   Point;
		////////		typedef CGAL::Straight_skeleton_2<K> Ss;
		////////		typedef boost::shared_ptr<Ss> SsPtr;
		////////		Polygon_2 poly;
		////////		poly.push_back(Point(-1, -1));
		////////		poly.push_back(Point(0, -12));
		////////		poly.push_back(Point(1, -1));
		////////		poly.push_back(Point(12, 0));
		////////		poly.push_back(Point(1, 1));
		////////		poly.push_back(Point(0, 12));
		////////		poly.push_back(Point(-1, 1));
		////////		poly.push_back(Point(-12, 0));
		////////		CRichModel model("sphere_half.obj");
		////////		model.LoadModel();
		////////		auto res = FillPlanarHolesWithAddingSkeletalPoints(model.m_Verts, model.m_Faces, model.GetNumOfVerts());
		////////		CBaseModel(res.first, res.second).SaveObjFile("sphere_out.obj");
		////////	}
	}

	void ConvertImplicitFunc2SurfaceMesh(function<CGAL::Surface_mesh_default_triangulation_3::Geom_traits::FT(CGAL::Surface_mesh_default_triangulation_3::Geom_traits::Point_3)> implicitFunc,
		CGAL::Surface_mesh_default_triangulation_3::Geom_traits::Point_3 center,
		double radius,
		double  angle, // 30 angular bound
		double gap, //0.1,  // radius bound
		double distance, //0.1); // distance bound
		CGAL::Surface_mesh<CGAL::Surface_mesh_default_triangulation_3::Geom_traits::Point_3>& sm)
	{
		// default triangulation for Surface_mesher
		typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
		// c2t3
		typedef CGAL::Complex_2_in_triangulation_3<Tr> C2t3;
		typedef Tr::Geom_traits GT;
		typedef GT::Sphere_3 Sphere_3;
		typedef GT::Point_3 Point_3;
		typedef GT::FT FT;
		typedef FT(*Function)(Point_3);
		typedef CGAL::Implicit_surface_3<GT, Function> Surface_3;
		typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
		Tr tr;            // 3D-Delaunay triangulation
		C2t3 c2t3(tr);   // 2D-complex in 3D-Delaunay triangulation
		// defining the surface
		Surface_3 surface(implicitFunc,             // pointer to function
			Sphere_3(center, radius)); // bounding sphere
		// Note that "2." above is the *squared* radius of the bounding sphere!
		// defining meshing criteria
		CGAL::Surface_mesh_default_criteria_3<Tr> criteria(angle,  // angular bound
			gap,  // radius bound
			distance); // distance bound
		// meshing surface
		CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Manifold_tag(), 20);
		CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, sm);
		//////////////{
		//////////auto func = [](CGAL::Surface_mesh_default_triangulation_3::Geom_traits::Point_3 p)->double
		//////////{
		//////////	double x2 = p.x() * p.x(), y2 = p.y() * p.y(), z2 = p.z() * p.z();
		//////////	//return x2 + y2 + z2 - 1;
		//////////	return 2 * p.y() * (y2 - 3 * x2) * (1 - z2) + (x2 + y2) * (x2 + y2) - (9 * z2 - 1) * (1 - z2);
		//////////	//return x2 + y2 - 1;
		//////////};
		//////////CGAL::Surface_mesh<CGAL::Surface_mesh_default_triangulation_3::Geom_traits::Point_3> sm;
		//////////ConvertImplicitFunc2SurfaceMesh(func,
		//////////	CGAL::ORIGIN,
		//////////	4,
		//////////	30,
		//////////	0.1,
		//////////	0.1,
		//////////	sm);
		//////////CGAL::IO::write_polygon_mesh("genus2.obj", sm);
		//////////////}
	}

	void ConvertImplicitFunc2SurfaceMesh(function<double(double, double, double)> implicitFunc_double,
		double x_center,
		double y_center,
		double z_center,
		double radius,
		double  angle, // 30 angular bound
		double gap, //0.1,  // radius bound
		double distance, //0.1); // distance bound
		CGAL::Surface_mesh<CGAL::Surface_mesh_default_triangulation_3::Geom_traits::Point_3>& sm)
	{
		auto implicitFunc
			= [&](CGAL::Surface_mesh_default_triangulation_3::Geom_traits::Point_3 pt)->CGAL::Surface_mesh_default_triangulation_3::Geom_traits::FT
		{
			return implicitFunc_double(pt.x(), pt.y(), pt.z());
		};
		CGAL::Surface_mesh_default_triangulation_3::Geom_traits::Point_3 center(x_center, y_center, z_center);
		ConvertImplicitFunc2SurfaceMesh(implicitFunc, center, radius, angle, gap, distance, sm);
	}

	template<typename InputIterator, typename Search_traits>
	void GetKdTree(InputIterator first, InputIterator beyond, CGAL::Kd_tree<Search_traits>& tree)
	{
		//see https://doc.cgal.org/latest/Spatial_searching/index.html
		tree.clear();
		tree.insert(first, beyond);
		{
			//////////vector<double_Kernel::Point_3> pts;
			//////////CGAL::Kd_tree<CGAL::Search_traits_3<double_Kernel>> tree;
			//////////GetKdTree(pts.begin(), pts.end(), tree);
			//////////or
			////////vector<double_Kernel::Point_2> pts;
			////////CGAL::Kd_tree<CGAL::Search_traits_2<double_Kernel>> tree;
			////////GetKdTree(pts.begin(), pts.end(), tree);
		}
	}


	template<typename Kernel>
	void SubdivideSurfaceMesh(CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm,
		int num, string method)
	{
		if (method == "Sqrt3")
		{
			CGAL::Subdivision_method_3::Sqrt3_subdivision(sm, CGAL::parameters::number_of_iterations(num));
		}
		else if (method == "CatmullClark")
		{
			CGAL::Subdivision_method_3::CatmullClark_subdivision(sm, CGAL::parameters::number_of_iterations(num));
		}
		else if (method == "Loop")
		{
			CGAL::Subdivision_method_3::Loop_subdivision(sm, CGAL::parameters::number_of_iterations(num));
		}
		else if (method == "DooSabin")
		{
			CGAL::Subdivision_method_3::DooSabin_subdivision(sm, CGAL::parameters::number_of_iterations(num));
		}
		else if (method == "PTQ")
		{
			typedef CGAL::Surface_mesh<CGAL::Point_3<Kernel>> Surface_mesh;
			class WLoop_mask_3 {
				typedef typename boost::graph_traits<Surface_mesh>::vertex_descriptor   vertex_descriptor;
				typedef typename boost::graph_traits<Surface_mesh>::halfedge_descriptor halfedge_descriptor;
				typedef typename boost::property_map<Surface_mesh, CGAL::vertex_point_t>::type Vertex_pmap;
				typedef typename boost::property_traits<Vertex_pmap>::value_type Point;
				typedef typename boost::property_traits<Vertex_pmap>::reference Point_ref;
				Surface_mesh& pmesh;
				Vertex_pmap vpm;
			public:
				WLoop_mask_3(Surface_mesh& pmesh)
					: pmesh(pmesh), vpm(get(CGAL::vertex_point, pmesh))
				{}
				void edge_node(halfedge_descriptor hd, Point& pt) {
					Point_ref p1 = get(vpm, target(hd, pmesh));
					Point_ref p2 = get(vpm, target(opposite(hd, pmesh), pmesh));
					Point_ref f1 = get(vpm, target(next(hd, pmesh), pmesh));
					Point_ref f2 = get(vpm, target(next(opposite(hd, pmesh), pmesh), pmesh));
					pt = Point((3 * (p1[0] + p2[0]) + f1[0] + f2[0]) / 8,
						(3 * (p1[1] + p2[1]) + f1[1] + f2[1]) / 8,
						(3 * (p1[2] + p2[2]) + f1[2] + f2[2]) / 8);
				}
				void vertex_node(vertex_descriptor vd, Point& pt) {
					double R[] = { 0.0, 0.0, 0.0 };
					Point_ref S = get(vpm, vd);
					std::size_t n = 0;
					for (halfedge_descriptor hd : halfedges_around_target(vd, pmesh)) {
						++n;
						Point_ref p = get(vpm, target(opposite(hd, pmesh), pmesh));
						R[0] += p[0];         R[1] += p[1];         R[2] += p[2];
					}
					if (n == 6) {
						pt = Point((10 * S[0] + R[0]) / 16, (10 * S[1] + R[1]) / 16, (10 * S[2] + R[2]) / 16);
					}
					else if (n == 3) {
						double B = (5.0 / 8.0 - std::sqrt(3 + 2 * std::cos(6.283 / n)) / 64.0) / n;
						double A = 1 - n * B;
						pt = Point((A * S[0] + B * R[0]), (A * S[1] + B * R[1]), (A * S[2] + B * R[2]));
					}
					else {
						double B = 3.0 / 8.0 / n;
						double A = 1 - n * B;
						pt = Point((A * S[0] + B * R[0]), (A * S[1] + B * R[1]), (A * S[2] + B * R[2]));
					}
				}
				void border_node(halfedge_descriptor hd, Point& ept, Point& vpt) {
					Point_ref ep1 = get(vpm, target(hd, pmesh));
					Point_ref ep2 = get(vpm, target(opposite(hd, pmesh), pmesh));
					ept = Point((ep1[0] + ep2[0]) / 2, (ep1[1] + ep2[1]) / 2, (ep1[2] + ep2[2]) / 2);
					CGAL::Halfedge_around_target_circulator<Surface_mesh> vcir(hd, pmesh);
					Point_ref vp1 = get(vpm, target(opposite(*vcir, pmesh), pmesh));
					Point_ref vp0 = get(vpm, target(*vcir, pmesh));
					--vcir;
					Point_ref vp_1 = get(vpm, target(opposite(*vcir, pmesh), pmesh));
					vpt = Point((vp_1[0] + 6 * vp0[0] + vp1[0]) / 8,
						(vp_1[1] + 6 * vp0[1] + vp1[1]) / 8,
						(vp_1[2] + 6 * vp0[2] + vp1[2]) / 8);
				}
			};
			CGAL::Subdivision_method_3::PTQ(sm,
				WLoop_mask_3(sm),
				CGAL::parameters::number_of_iterations(num));
		}
		////////////////{
		////////////////	CGAL::IO::read_polygon_mesh("torus.obj", sm);
		////////////////	SubdivideSurfaceMesh(sm, 2, "DooSabin"); //recommend: PTQ, Sqrt3, Loop
		////////////////	CGAL::IO::write_polygon_mesh("torus2.obj", sm);
		////////////////}
	}

	template<typename Kernel>
	void SimplifySurfaceMesh(CGAL::Surface_mesh<CGAL::Point_3<Kernel>>& sm,
		double ratio, //No. new faces / No. old faces = e.g., 0.1
		string GarlandHeckbertPolicy)
	{
		namespace SMS = CGAL::Surface_mesh_simplification;
		if (GarlandHeckbertPolicy == "Classic_plane")
		{
			typedef CGAL::Surface_mesh<CGAL::Point_3<Kernel>> Surface_mesh;
			typedef SMS::GarlandHeckbert_plane_policies<Surface_mesh, Kernel> GHPolicies;
			SMS::Count_ratio_stop_predicate<Surface_mesh> stop(ratio);
			typedef typename GHPolicies::Get_cost                                        GH_cost;
			typedef typename GHPolicies::Get_placement                                   GH_placement;
			typedef SMS::Bounded_normal_change_placement<GH_placement>                    Bounded_GH_placement;
			GHPolicies gh_policies(sm);
			const GH_cost& gh_cost = gh_policies.get_cost();
			const GH_placement& gh_placement = gh_policies.get_placement();
			Bounded_GH_placement placement(gh_placement);
			int r = SMS::edge_collapse(sm, stop,
				CGAL::parameters::get_cost(gh_cost)
				.get_placement(placement));
		}
		else if (GarlandHeckbertPolicy == "Prob_plane")
		{
			typedef CGAL::Surface_mesh<CGAL::Point_3<Kernel>> Surface_mesh;
			typedef SMS::GarlandHeckbert_probabilistic_plane_policies<Surface_mesh, Kernel> GHPolicies;
			SMS::Count_ratio_stop_predicate<Surface_mesh> stop(ratio);
			typedef typename GHPolicies::Get_cost                                        GH_cost;
			typedef typename GHPolicies::Get_placement                                   GH_placement;
			typedef SMS::Bounded_normal_change_placement<GH_placement>                    Bounded_GH_placement;
			GHPolicies gh_policies(sm);
			const GH_cost& gh_cost = gh_policies.get_cost();
			const GH_placement& gh_placement = gh_policies.get_placement();
			Bounded_GH_placement placement(gh_placement);
			int r = SMS::edge_collapse(sm, stop,
				CGAL::parameters::get_cost(gh_cost)
				.get_placement(placement));
		}
		else if (GarlandHeckbertPolicy == "Classic_tri")
		{
			typedef CGAL::Surface_mesh<CGAL::Point_3<Kernel>> Surface_mesh;
			typedef SMS::GarlandHeckbert_triangle_policies<Surface_mesh, Kernel>  GHPolicies;
			SMS::Count_ratio_stop_predicate<Surface_mesh> stop(ratio);
			typedef typename GHPolicies::Get_cost                                        GH_cost;
			typedef typename GHPolicies::Get_placement                                   GH_placement;
			typedef SMS::Bounded_normal_change_placement<GH_placement>                    Bounded_GH_placement;
			GHPolicies gh_policies(sm);
			const GH_cost& gh_cost = gh_policies.get_cost();
			const GH_placement& gh_placement = gh_policies.get_placement();
			Bounded_GH_placement placement(gh_placement);
			int r = SMS::edge_collapse(sm, stop,
				CGAL::parameters::get_cost(gh_cost)
				.get_placement(placement));
		}
		else if (GarlandHeckbertPolicy == "Prob_tri")
		{
			typedef CGAL::Surface_mesh<CGAL::Point_3<Kernel>> Surface_mesh;
			typedef SMS::GarlandHeckbert_probabilistic_triangle_policies<Surface_mesh, Kernel>  GHPolicies;
			SMS::Count_ratio_stop_predicate<Surface_mesh> stop(ratio);
			typedef typename GHPolicies::Get_cost                                        GH_cost;
			typedef typename GHPolicies::Get_placement                                   GH_placement;
			typedef SMS::Bounded_normal_change_placement<GH_placement>                    Bounded_GH_placement;
			GHPolicies gh_policies(sm);
			const GH_cost& gh_cost = gh_policies.get_cost();
			const GH_placement& gh_placement = gh_policies.get_placement();
			Bounded_GH_placement placement(gh_placement);
			int r = SMS::edge_collapse(sm, stop,
				CGAL::parameters::get_cost(gh_cost)
				.get_placement(placement));
		}
	}
}