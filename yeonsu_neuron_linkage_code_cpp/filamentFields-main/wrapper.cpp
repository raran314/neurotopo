#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "filamentFields.h"

namespace py = pybind11;

PYBIND11_MODULE(filamentFields, m) {
    // std::cout << "Filament Processing module initialized" << std::endl;
    py::class_<filamentFields>(m, "filamentFields")
        .def(py::init<const std::vector<Eigen::MatrixXd>&>())
        .def(py::init<const std::vector<Eigen::MatrixXd>&, const Eigen::MatrixXd&>())
        .def("update_filament_nodes_list", &filamentFields::update_filament_nodes_list)
        .def("update_contact_array", &filamentFields::update_contact_array)
        .def("return_filament_nodes_list", &filamentFields::return_filament_nodes_list)
        .def("return_filament_edges_list", &filamentFields::return_filament_edges_list)
        .def("return_number_of_labels", &filamentFields::return_number_of_labels)
        .def("return_volume_fraction", &filamentFields::return_volume_fraction)
        .def("return_orientational_order_parameter", &filamentFields::return_orientational_order_parameter)
        .def("return_local_Q_tensor", &filamentFields::return_local_Q_tensor)
        .def("return_entanglement", &filamentFields::return_entanglement)
        .def("return_total_entanglement", &filamentFields::return_total_entanglement)
        .def("return_number_of_local_contacts", &filamentFields::return_number_of_local_contacts)
        .def("return_force_sum", &filamentFields::return_force_sum)
        .def("return_all_nodes", &filamentFields::return_all_nodes)
        .def("return_all_edges", &filamentFields::return_all_edges)
        .def("return_node_labels", &filamentFields::return_node_labels)
        .def("return_edge_labels", &filamentFields::return_edge_labels)
        .def("return_total_linking_matrix", &filamentFields::return_total_linking_matrix)
        .def("compute_total_linking_matrix", &filamentFields::compute_total_linking_matrix)
        .def("precompute", &filamentFields::precompute)
        .def("sample_edges_locally", &filamentFields::sample_edges_locally)
        .def("analyze_local_volume", &filamentFields::analyze_local_volume)
        .def("analyze_local_volume_from_precomputed", &filamentFields::analyze_local_volume_from_precomputed)
        .def("analyze_local_volume_over_domain", &filamentFields::analyze_local_volume_over_domain);
        
}