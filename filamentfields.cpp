#define PYBIND11_DETAILED_ERROR_MESSAGES

#include "filamentFields.h"
#include <iostream>

filamentFields::filamentFields(const std::vector<Eigen::MatrixXd>& filament_nodes_list) :
    filament_nodes_list(filament_nodes_list)

{
    get_all_nodes();
    get_all_edges();
    get_node_labels();
    get_edge_labels();
}

void filamentFields::get_all_nodes() {
    for (const Eigen::MatrixXd& nodes : filament_nodes_list) {
        all_nodes.conservativeResize(all_nodes.rows() + nodes.rows(), 3);
        all_nodes.bottomRows(nodes.rows()) = nodes;
    }
}

void filamentFields::get_all_edges() {
    filament_edges_list.clear();
    all_edges.resize(0, 6); // Ensure all_edges is an Nx6 matrix

    for (const Eigen::MatrixXd& nodes : filament_nodes_list) {
        Eigen::MatrixXd edges(nodes.rows() - 1, 6);

        for (int idx = 0; idx < nodes.rows() - 1; ++idx) {
            edges.row(idx).segment<3>(0) = nodes.row(idx);      // Start point of the edge
            edges.row(idx).segment<3>(3) = nodes.row(idx + 1);  // End point of the edge
        }
        filament_edges_list.push_back(edges);
    }

    for (const Eigen::MatrixXd& edges : filament_edges_list) {
        all_edges.conservativeResize(all_edges.rows() + edges.rows(), 6);
        all_edges.bottomRows(edges.rows()) = edges;
    }
}


void filamentFields::get_node_labels() {
    node_labels = Eigen::VectorXi::Zero(all_nodes.rows());
    int label = 0;
    int cursor = 0;
    for (const Eigen::MatrixXd& nodes : filament_nodes_list) {
        node_labels.segment(cursor, nodes.rows()).setConstant(label);
        cursor += nodes.rows();
        label += 1;
    }
}

void filamentFields::get_edge_labels() {
    edge_labels = Eigen::VectorXi::Zero(all_edges.rows());
    int label = 0;
    int cursor = 0;
    for (const Eigen::MatrixXd& edges : filament_edges_list) {
        edge_labels.segment(cursor, edges.rows()).setConstant(label);
        cursor += edges.rows();
        label += 1;
    }
}

Eigen::VectorXi filamentFields::sample_edges_locally(const Eigen::Vector3d& query_point, double R_omega) const {
    Eigen::VectorXi local_edge_labels = Eigen::VectorXi::Zero(all_edges.rows());
    for (int idx = 0; idx < all_edges.rows(); ++idx) {
        Eigen::Vector3d edge_start = all_edges.row(idx).segment<3>(0);
        Eigen::Vector3d edge_end = all_edges.row(idx).segment<3>(3);

        if (((edge_start - query_point).norm() < R_omega) && ((edge_end - query_point).norm() < R_omega)) {
            local_edge_labels(idx) = 1;
        }

    }
    return local_edge_labels;
}

Eigen::MatrixXd filamentFields::analyzeLocalVolume(const Eigen::Vector3d& query_point, double R_omega, double rod_radius) {
    // Sample edges locally
    Eigen::VectorXi local_edge_labels = sample_edges_locally(query_point, R_omega);
    number_of_labels = 0;
    // Count the number of local edges
    int local_edge_count = local_edge_labels.sum();
    if (local_edge_count == 0) {
        number_of_labels = 0;
        volume_fraction = std::numeric_limits<double>::quiet_NaN();
        orientational_order_parameter = std::numeric_limits<double>::quiet_NaN();
        entanglement = std::numeric_limits<double>::quiet_NaN();
        return Eigen::MatrixXd(0, 6);
    }

    // Extract local edges
    Eigen::MatrixXd local_edges(local_edge_count, 6);
    Eigen::VectorXi local_edge_indices = Eigen::VectorXi::Zero(local_edge_count);
    int index = 0;
    for (int i = 0; i < all_edges.rows(); ++i) {
        if (local_edge_labels(i) == 1) {
            local_edge_indices(index) = i;
            local_edges.row(index++) = all_edges.row(i);
        }
    }
    // Unique labels
    std::vector<int> unique_labels;
    for (int i = 0; i < local_edge_count; ++i) {
        if (std::find(unique_labels.begin(), unique_labels.end(), local_edge_labels(i)) == unique_labels.end()) {
            unique_labels.push_back(local_edge_labels(i));
            number_of_labels++;
        }
    }

    double edge_length_sum = 0.0;
    for (int i = 0; i < local_edge_count; ++i) {
        Eigen::Vector3d edge_start = local_edges.row(i).segment<3>(0);
        Eigen::Vector3d edge_end = local_edges.row(i).segment<3>(3);
        edge_length_sum += (edge_end - edge_start).norm();
    }

    // volume fraction
    volume_fraction = (M_PI * rod_radius * rod_radius * edge_length_sum) / (4.0 / 3.0 * M_PI * R_omega * R_omega * R_omega);

    // Orientational order parameter
    Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
    for (int i = 0; i < local_edge_count; ++i) {
        Eigen::Vector3d edge_start = local_edges.row(i).segment<3>(0);
        Eigen::Vector3d edge_end = local_edges.row(i).segment<3>(3);
        Eigen::Vector3d edge = edge_end - edge_start;
        edge.normalize();
        Q += (3.0 * edge * edge.transpose() - Eigen::Matrix3d::Identity())/2.0;
    }
    Q /= local_edge_count;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(Q);
    Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
    // Find maximum absolute eigenvalue
    double max_eigenvalue = eigenvalues.cwiseAbs().maxCoeff();
    orientational_order_parameter = max_eigenvalue;

    // Entanglement
    Eigen::MatrixXd entanglement_matrix(local_edge_count, local_edge_count);
    entanglement_matrix.setConstant(std::numeric_limits<double>::quiet_NaN());
    filamentFields::compute_edge_wise_entanglement(local_edges, local_edge_labels, entanglement_matrix);
    // ignore nan values and take absolute value and sum
    // double entanglement = entanglement_matrix.unaryExpr([](double x) { return std::abs(x); }).sum();
    entanglement = entanglement_matrix.unaryExpr([](double x) {
        return std::isnan(x) ? 0.0 : std::abs(x);
    }).sum();

    return local_edges;
}



double filamentFields::_clip(double x, double lower, double upper) const {
    return std::min(std::max(x, lower), upper);
}

double filamentFields::compute_linking_number_for_edges(const Eigen::VectorXd& e_i, const Eigen::VectorXd& e_j) const {
    Eigen::Vector3d r_ij = e_i.segment<3>(0) - e_j.segment<3>(0);
    Eigen::Vector3d r_ijj = e_i.segment<3>(0) - e_j.segment<3>(3);
    Eigen::Vector3d r_iij = e_i.segment<3>(3) - e_j.segment<3>(0);
    Eigen::Vector3d r_iijj = e_i.segment<3>(3) - e_j.segment<3>(3);

    double tol = 1e-6;
    
    Eigen::Vector3d n1 = r_ij.cross(r_ijj);
    n1 /= (n1.norm() + tol);
    
    Eigen::Vector3d n2 = r_ijj.cross(r_iijj);
    n2 /= (n2.norm() + tol);
    
    Eigen::Vector3d n3 = r_iijj.cross(r_iij);
    n3 /= (n3.norm() + tol);
    
    Eigen::Vector3d n4 = r_iij.cross(r_ij);
    n4 /= (n4.norm() + tol);

    return -1.0 / (4 * M_PI) * std::abs(
        std::asin(filamentFields::_clip(n1.dot(n2), -1.0 + tol, 1.0 - tol)) +
        std::asin(filamentFields::_clip(n2.dot(n3), -1.0 + tol, 1.0 - tol)) +
        std::asin(filamentFields::_clip(n3.dot(n4), -1.0 + tol, 1.0 - tol)) +
        std::asin(filamentFields::_clip(n4.dot(n1), -1.0 + tol, 1.0 - tol))
    );
}



void filamentFields::compute_edge_wise_entanglement(const Eigen::MatrixXd& _all_edges, const Eigen::VectorXi& labels, Eigen::MatrixXd& entanglement_matrix) const {
    int num_all_edges = _all_edges.rows();
    for (int idx = 0; idx < num_all_edges; ++idx) {
        const Eigen::VectorXd edge1 = _all_edges.row(idx);

        for (int jdx = idx + 1; jdx < num_all_edges; ++jdx) {
            if (labels(idx) == labels(jdx)) {
                continue;
            }

            const Eigen::VectorXd edge2 = _all_edges.row(jdx);
            // entanglement_matrix(idx, jdx) = filamentFields::compute_linking_number_for_edges(edge1, edge2);
            double lk = filamentFields::compute_linking_number_for_edges(edge1, edge2);
            entanglement_matrix(idx, jdx) = lk;
        }
    }
}
