#pragma once
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Polygon_2.h>
#include <list>
#include <queue>
#include <vector>
#include <set>
#include <map>
using namespace std;

namespace Model3D
{
	struct FaceInfo2
	{
		FaceInfo2() {}
		int nesting_level;
		bool in_domain() {
			return nesting_level % 2 == 1;
		}
	};
	typedef CGAL::Exact_predicates_inexact_constructions_kernel CDT_Kernel;
	typedef CGAL::Triangulation_vertex_base_2<CDT_Kernel>                      CDT_Vb;
	typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2, CDT_Kernel>    CDT_Fbb;
	typedef CGAL::Constrained_triangulation_face_base_2<CDT_Kernel, CDT_Fbb>        CDT_Fb;
	typedef CGAL::Triangulation_data_structure_2<CDT_Vb, CDT_Fb>               CDT_TDS;
	typedef CGAL::Exact_predicates_tag                                CDT_Itag;
	typedef CGAL::Constrained_Delaunay_triangulation_2<CDT_Kernel, CDT_TDS, CDT_Itag>  CDT;

	void	mark_domains(CDT& ct,
		CDT::Face_handle start,
		int index,
		std::list<CDT::Edge>& border);
	void mark_domains(CDT& cdt);
}