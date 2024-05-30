#define PYBIND11_DETAILED_ERROR_MESSAGES

#include "filamentFields.h"
#include <iostream>
#include <armadillo>

filamentFields::filamentFields(const std::vector<arma::mat>& filament_nodes_list) :
    filament_nodes_list(filament_nodes_list)
{
    get_all_nodes();
    get_all_edges();
    get_node_labels();
    get_edge_labels();
}

void filamentFields::get_all_nodes() {
    for (const arma::mat& nodes : filament_nodes_list) {
        arma::mat resizedNodes = arma::zeros(nodes.n_rows, 3);
        resizedNodes.cols(0, 2) = nodes.cols(0, 2);
        all_nodes = arma::join_cols(all_nodes, resizedNodes);
    }
}

void filamentFields::get_all_edges() {
    filament_edges_list.clear();
    all_edges = arma::zeros(0, 6); // Ensure all_edges is an Nx6 matrix

    for (const arma::mat& nodes : filament_nodes_list) {
        arma::mat edges(nodes.n_rows - 1, 6);

        for (int idx = 0; idx < nodes.n_rows - 1; ++idx) {
            edges(idx, arma::span(0, 2)) = nodes.row(idx);      // Start point of the edge
            edges(idx, arma::span(3, 5)) = nodes.row(idx + 1);  // End point of the edge
        }
        filament_edges_list.push_back(edges);
    }

    for (const arma::mat& edges : filament_edges_list) {
        all_edges = arma::join_cols(all_edges, edges);
    }
}

void filamentFields::get_node_labels() {
    node_labels = arma::zeros<arma::ivec>(all_nodes.n_rows);
    int label = 0;
    int cursor = 0;
    for (const arma::mat& nodes : filament_nodes_list) {
        node_labels.subvec(cursor, cursor + nodes.n_rows - 1).fill(label);
        cursor += nodes.n_rows;
        label += 1;
    }
}

void filamentFields::get_edge_labels() {
    edge_labels = arma::zeros<arma::ivec>(all_edges.n_rows);
    int label = 0;
    int cursor = 0;
    for (const arma::mat& edges : filament_edges_list) {
        edge_labels.subvec(cursor, cursor + edges.n_rows - 1).fill(label);
        cursor += edges.n_rows;
        label += 1;
    }
}

arma::ivec filamentFields::sample_edges_locally(const arma::vec& query_point, double R_omega) const {
    arma::ivec local_edge_labels = arma::zeros<arma::ivec>(all_edges.n_rows);
    for (int idx = 0; idx < all_edges.n_rows; ++idx) {
        arma::vec edge_start = all_edges.row(idx).subvec(0, 2);
        arma::vec edge_end = all_edges.row(idx).subvec(3, 5);

        if (((edge_start - query_point).norm() < R_omega) && ((edge_end - query_point).norm() < R_omega)) {
            local_edge_labels(idx) = 1;
        }
    }
    return local_edge_labels;
}

arma::mat filamentFields::analyzeLocalVolume(const arma::vec& query_point, double R_omega, double rod_radius) {
    // Sample edges locally
    arma::ivec local_edge_labels = sample_edges_locally(query_point, R_omega);
    number_of_labels = 0;
    // Count the number of local edges
    int local_edge_count = arma::accu(local_edge_labels);
    if (local_edge_count == 0) {
        number_of_labels = 0;
        volume_fraction = std::numeric_limits<double>::quiet_NaN();
        orientational_order_parameter = std::numeric_limits<double>::quiet_NaN();
        entanglement = std::numeric_limits<double>::quiet_NaN();
        return arma::mat(0, 6);
    }

    // Extract local edges
    arma::mat local_edges(local_edge_count, 6);
    arma::ivec local_edge_indices = arma::zeros<arma::ivec>(local_edge_count);
    int index = 0;
    for (int i = 0; i < all_edges.n_rows; ++i) {
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
        arma::vec edge_start = local_edges.row(i).subvec(0, 2);
        arma::vec edge_end = local_edges.row(i).subvec(3, 5);
        edge_length_sum += arma::norm(edge_end - edge_start);
    }

    // volume fraction
    volume_fraction = (M_PI * rod_radius * rod_radius * edge_length_sum) / (4.0 / 3.0 * M_PI * R_omega * R_omega * R_omega);

    // Orientational order parameter
    arma::mat Q = arma::zeros(3, 3);
    for (int i = 0; i < local_edge_count; ++i) {
        arma::vec edge_start = local_edges.row(i).subvec(0, 2);
        arma::vec edge_end = local_edges.row(i).subvec(3, 5);
        arma::vec edge = edge_end - edge_start;
        edge /= arma::norm(edge);
        Q += (3.0 * edge * edge.t() - arma::eye<arma::mat>(3, 3)) / 2.0;
    }
    Q /= local_edge_count;
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, Q);
    double max_eigenvalue = arma::max(arma::abs(eigval));
    orientational_order_parameter = max_eigenvalue;

    // Entanglement
    arma::mat entanglement_matrix(local_edge_count, local_edge_count, arma::fill::zeros);
    compute_edge_wise_entanglement(local_edges, local_edge_labels, entanglement_matrix);
    entanglement = arma::accu(arma::abs(entanglement_matrix));

    return local_edges;
}

double filamentFields::_clip(double x, double lower, double upper) const {
    return std::min(std::max(x, lower), upper);
}

double filamentFields::compute_linking_number_for_edges(const
