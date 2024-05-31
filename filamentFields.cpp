#define PYBIND11_DETAILED_ERROR_MESSAGES

#include "filamentFields.h"
#include <iostream>

filamentFields::filamentFields(const std::vector<arma::mat>& filament_nodes_list) :
    filament_nodes_list(filament_nodes_list) {

    get_all_nodes();
    get_all_edges();
    get_node_labels();
    get_edge_labels();
}

void filamentFields::get_all_nodes() {
    for (const arma::mat& nodes : filament_nodes_list) {
        all_nodes.resize(all_nodes.n_rows + nodes.n_rows, 3);
        all_nodes.tail_rows(nodes.n_rows) = nodes;
    }
}

void filamentFields::get_all_edges() {
    filament_edges_list.clear();
    all_edges.resize(0, 6);

    for (const arma::mat& nodes : filament_nodes_list) {
        arma::mat edges(nodes.n_rows - 1, 6);

        for (size_t idx = 0; idx < nodes.n_rows - 1; ++idx) {
            edges.row(idx).cols(0, 2) = nodes.row(idx);
            edges.row(idx).cols(3, 5) = nodes.row(idx + 1);
        }
        filament_edges_list.push_back(edges);
    }

    for (const arma::mat& edges : filament_edges_list) {
        all_edges.resize(all_edges.n_rows + edges.n_rows, 6);
        all_edges.tail_rows(edges.n_rows) = edges;
    }
}

void filamentFields::get_node_labels() {
    node_labels = arma::vec(all_nodes.n_rows, arma::fill::zeros);
    int label = 0;
    int cursor = 0;
    for (const arma::mat& nodes : filament_nodes_list) {
        node_labels.subvec(cursor, cursor + nodes.n_rows - 1).fill(label);
        cursor += nodes.n_rows;
        label += 1;
    }
}

void filamentFields::get_edge_labels() {
    edge_labels = arma::vec(all_edges.n_rows, arma::fill::zeros);
    int label = 0;
    int cursor = 0;
    for (const arma::mat& edges : filament_edges_list) {
        edge_labels.subvec(cursor, cursor + edges.n_rows - 1).fill(label);
        cursor += edges.n_rows;
        label += 1;
    }
}

arma::vec filamentFields::sample_edges_locally(const arma::vec& query_point, double R_omega) const {
    arma::vec local_edge_labels = arma::vec(all_edges.n_rows, arma::fill::zeros);
    for (size_t idx = 0; idx < all_edges.n_rows; ++idx) {
        arma::vec edge_start = all_edges.row(idx).cols(0, 2).t();
        arma::vec edge_end = all_edges.row(idx).cols(3, 5).t();

        if (arma::norm(edge_start - query_point) < R_omega && arma::norm(edge_end - query_point) < R_omega) {
            local_edge_labels(idx) = 1;
        }
    }
    return local_edge_labels;
}

arma::mat filamentFields::analyzeLocalVolume(const arma::vec& query_point, double R_omega, double rod_radius) {
    arma::vec local_edge_labels = sample_edges_locally(query_point, R_omega);
    number_of_labels = 0;
    int local_edge_count = arma::sum(local_edge_labels);
    if (local_edge_count == 0) {
        number_of_labels = 0;
        volume_fraction = std::numeric_limits<double>::quiet_NaN();
        orientational_order_parameter = std::numeric_limits<double>::quiet_NaN();
        entanglement = std::numeric_limits<double>::quiet_NaN();
        return arma::mat(0, 6);
    }

    arma::mat local_edges(local_edge_count, 6);
    arma::vec local_edge_indices = arma::vec(local_edge_count, arma::fill::zeros);
    int index = 0;
    for (size_t i = 0; i < all_edges.n_rows; ++i) {
        if (local_edge_labels(i) == 1) {
            local_edge_indices(index) = i;
            local_edges.row(index++) = all_edges.row(i);
        }
    }

    std::vector<int> unique_labels;
    for (int i = 0; i < local_edge_count; ++i) {
        if (std::find(unique_labels.begin(), unique_labels.end(), local_edge_labels(i)) == unique_labels.end()) {
            unique_labels.push_back(local_edge_labels(i));
            number_of_labels++;
        }
    }

    double edge_length_sum = 0.0;
    for (int i = 0; i < local_edge_count; ++i) {
        arma::vec edge_start = local_edges.row(i).cols(0, 2).t();
        arma::vec edge_end = local_edges.row(i).cols(3, 5).t();
        edge_length_sum += arma::norm(edge_end - edge_start);
    }

    volume_fraction = (M_PI * rod_radius * rod_radius * edge_length_sum) / (4.0 / 3.0 * M_PI * R_omega * R_omega * R_omega);

    arma::mat Q = arma::mat(3, 3, arma::fill::zeros);
    for (int i = 0; i < local_edge_count; ++i) {
        arma::vec edge_start = local_edges.row(i).cols(0, 2).t();
        arma::vec edge_end = local_edges.row(i).cols(3, 5).t();
        arma::vec edge = edge_end - edge_start;
        edge /= arma::norm(edge);
        Q += (3.0 * edge * edge.t() - arma::mat(3, 3, arma::fill::eye)) / 2.0;
    }
    Q /= local_edge_count;
    arma::vec eigenvalues = arma::eig_sym(Q);
    double max_eigenvalue = arma::abs(eigenvalues).max();
    orientational_order_parameter = max_eigenvalue;

    arma::mat entanglement_matrix(local_edge_count, local_edge_count, arma::fill::nan);
    filamentFields::compute_edge_wise_entanglement(local_edges, local_edge_labels, entanglement_matrix);
    entanglement = arma::accu(entanglement_matrix.transform([](double val) { return std::isnan(val) ? 0.0 : std::abs(val); }));

    return local_edges;
}

double filamentFields::_clip(double x, double lower, double upper) const {
    return std::min(std::max(x, lower), upper);
}

double filamentFields::compute_linking_number_for_edges(const arma::rowvec& e_i, const arma::rowvec& e_j) const {
    arma::vec r_ij = e_i.cols(0, 2).t() - e_j.cols(0, 2).t();
    arma::vec r_ijj = e_i.cols(0, 2).t() - e_j.cols(3, 5).t();
    arma::vec r_iij = e_i.cols(3, 5).t() - e_j.cols(0, 2).t();
    arma::vec r_iijj = e_i.cols(3, 5).t() - e_j.cols(3, 5).t();

    double tol = 1e-6;

    arma::vec n1 = arma::cross(r_ij, r_ijj);
    n1 /= (arma::norm(n1) + tol);

    arma::vec n2 = arma::cross(r_ijj, r_iijj);
    n2 /= (arma::norm(n2) + tol);

    arma::vec n3 = arma::cross(r_iijj, r_iij);
    n3 /= (arma::norm(n3) + tol);

    arma::vec n4 = arma::cross(r_iij, r_ij);
    n4 /= (arma::norm(n4) + tol);

    return -1.0 / (4 * M_PI) * std::abs(
        std::asin(filamentFields::_clip(arma::dot(n1, n2), -1.0 + tol, 1.0 - tol)) +
        std::asin(filamentFields::_clip(arma::dot(n2, n3), -1.0 + tol, 1.0 - tol)) +
        std::asin(filamentFields::_clip(arma::dot(n3, n4), -1.0 + tol, 1.0 - tol)) +
        std::asin(filamentFields::_clip(arma::dot(n4, n1), -1.0 + tol, 1.0 - tol))
    );
}

void filamentFields::compute_edge_wise_entanglement(const arma::mat& _all_edges, const arma::vec& labels, arma::mat& entanglement_matrix) const {
    size_t num_all_edges = _all_edges.n_rows;
    for (size_t idx = 0; idx < num_all_edges; ++idx) {
        const arma::rowvec edge1 = _all_edges.row(idx);

        for (size_t jdx = idx + 1; jdx < num_all_edges; ++jdx) {
            if (labels(idx) == labels(jdx)) {
                continue;
            }

            const arma::rowvec edge2 = _all_edges.row(jdx);
            double lk = filamentFields::compute_linking_number_for_edges(edge1, edge2);
            entanglement_matrix(idx, jdx) = lk;
        }
    }
}
