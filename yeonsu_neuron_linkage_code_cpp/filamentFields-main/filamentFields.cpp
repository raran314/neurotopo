#define PYBIND11_DETAILED_ERROR_MESSAGES
#define INV_TWO_PI                                                             \
  (0.1591549430918953357688837633725143620344596457404564487476673440)

#include "filamentFields.h"
#include <iostream>
#include <algorithm>
// #include <cmath>

filamentFields::filamentFields(const std::vector<Eigen::MatrixXd>& filament_nodes_list) :
    filament_nodes_list(filament_nodes_list)

{
    number_of_total_contacts = 0;
    number_of_local_contacts = 0;
    force_sum = 0;
    total_entanglement = 0;
    Q_tensors.clear();
    Q_tensors.reserve(all_edges.rows());

    get_all_nodes();
    get_all_edges();
    get_node_labels();
    get_edge_labels();
    // compute_total_linking_matrix();
}

filamentFields::filamentFields(const std::vector<Eigen::MatrixXd>& filament_nodes_list, const Eigen::MatrixXd& contact_array) :
    filament_nodes_list(filament_nodes_list), contact_array(contact_array)
{
    number_of_total_contacts = contact_array.rows();
    number_of_local_contacts = 0;
    force_sum = 0;
    total_entanglement = 0;
    Q_tensors.clear();
    Q_tensors.reserve(all_edges.rows());

    get_all_nodes();
    get_all_edges();
    get_node_labels();
    get_edge_labels();
    // compute_total_linking_matrix();
}

void filamentFields::update_filament_nodes_list(const std::vector<Eigen::MatrixXd>& _filament_nodes_list)
{
    filament_nodes_list = _filament_nodes_list;
    total_entanglement = 0;
    is_precomputed = false;
    Q_tensors.clear();
    Q_tensors.reserve(all_edges.rows());

    get_all_nodes();
    get_all_edges();
    get_node_labels();
    get_edge_labels();
    // compute_total_linking_matrix();
}

void filamentFields::update_contact_array(const Eigen::MatrixXd& _contact_array)
{
    contact_array = _contact_array;
    number_of_total_contacts = contact_array.rows();
    number_of_local_contacts = 0;
    force_sum = 0;
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

void filamentFields::get_edge_pairs(double R_omega) {
    edge_pairs.clear();
    edge_pairs.reserve(all_edges.rows());

    edge_lengths = Eigen::VectorXd::Zero(all_edges.rows());
    for (int idx = 0; idx < all_edges.rows(); ++idx) {
        Eigen::Vector3d edge1_start = all_edges.row(idx).segment<3>(0);
        Eigen::Vector3d edge1_end = all_edges.row(idx).segment<3>(3);
        edge_lengths(idx) = (edge1_end - edge1_start).norm();

        for (int jdx = idx + 1; jdx < all_edges.rows(); ++jdx) {
            if (edge_labels(idx) == edge_labels(jdx)) {
                continue;
            }

            Eigen::Vector3d edge2_start = all_edges.row(jdx).segment<3>(0);
            Eigen::Vector3d edge2_end = all_edges.row(jdx).segment<3>(3);

            double dist1 = (edge1_start - edge2_start).norm();
            double dist2 = (edge1_start - edge2_end).norm();
            double dist3 = (edge1_end - edge2_start).norm();
            double dist4 = (edge1_end - edge2_end).norm();
            if (dist1 < R_omega && dist2 < R_omega && dist3 < R_omega && dist4 < R_omega) {
                edge_pairs.push_back(std::make_pair(idx, jdx));
            }
        }
    }

}

void filamentFields::precompute(double R_omega) {
    // compute_all_edge_lengths(); 
    if (is_precomputed) {
        return;
    }

    get_edge_pairs(R_omega); // comes first to compute edge pairs
    compute_total_linking_matrix();
    compute_all_Q_tensors();
    is_precomputed = true;
}

void filamentFields::compute_total_linking_matrix() {
    int num_edges = all_edges.rows();
    total_linking_matrix = Eigen::MatrixXd::Zero(num_edges, num_edges);
    total_linking_matrix.setConstant(std::numeric_limits<double>::quiet_NaN());

    // filamentFields::compute_edge_wise_entanglement(all_edges, edge_labels, total_linking_matrix);
    for (auto& pair : edge_pairs) {
        int idx = pair.first;
        int jdx = pair.second;
        if (edge_labels(idx) == edge_labels(jdx)) {
            continue;
        }
        const Eigen::VectorXd edge1 = all_edges.row(idx);
        const Eigen::VectorXd edge2 = all_edges.row(jdx);
        double lk = filamentFields::compute_linking_number_for_edges(edge1, edge2);
        total_linking_matrix(idx, jdx) = lk;
    }

    total_entanglement = total_linking_matrix.unaryExpr([](double x) -> double {
        return std::isnan(x) ? 0.0 : std::abs(x);
    }).sum();
}

void filamentFields::compute_all_Q_tensors() {
    Q_tensors.clear();
    int num_edges = all_edges.rows();
    Eigen::MatrixXd Q_tensor = Eigen::MatrixXd::Zero(3, 3);
    Eigen::MatrixXd Q_tensor_sum = Eigen::MatrixXd::Zero(3, 3);
    for (int idx = 0; idx < num_edges; ++idx) {
        Eigen::Vector3d edge_start = all_edges.row(idx).segment<3>(0);
        Eigen::Vector3d edge_end = all_edges.row(idx).segment<3>(3);
        Eigen::Vector3d edge = edge_end - edge_start;
        edge.normalize();
        Q_tensor = (3.0 * edge * edge.transpose() - Eigen::Matrix3d::Identity())/2.0;
        Q_tensors.push_back(Q_tensor);
    }
}

void filamentFields::compute_all_edge_lengths() {
    edge_lengths = Eigen::VectorXd::Zero(all_edges.rows());
    for (int idx = 0; idx < all_edges.rows(); ++idx) {
        Eigen::Vector3d edge_start = all_edges.row(idx).segment<3>(0);
        Eigen::Vector3d edge_end = all_edges.row(idx).segment<3>(3);
        edge_lengths(idx) = (edge_end - edge_start).norm();
    }
}

Eigen::VectorXi filamentFields::sample_edges_locally(const Eigen::Vector3d& query_point, double R_omega) const {
    Eigen::VectorXi local_edge_trues = Eigen::VectorXi::Zero(all_edges.rows());
    for (int idx = 0; idx < all_edges.rows(); ++idx) {
        Eigen::Vector3d edge_start = all_edges.row(idx).segment<3>(0);
        Eigen::Vector3d edge_end = all_edges.row(idx).segment<3>(3);

        if (((edge_start - query_point).norm() < R_omega) && ((edge_end - query_point).norm() < R_omega)) {
            local_edge_trues(idx) = 1;
        }

    }
    return local_edge_trues;
}

Eigen::MatrixXd filamentFields::analyze_local_volume(const Eigen::Vector3d& query_point, double R_omega, double rod_radius) {
    // Sample edges locally
    Eigen::VectorXi local_edge_trues = sample_edges_locally(query_point, R_omega);
    number_of_labels = 0;
    // Count the number of local edges
    int local_edge_count = local_edge_trues.sum();
    if (local_edge_count == 0) {
        number_of_labels = 0;
        volume_fraction = std::numeric_limits<double>::quiet_NaN();
        orientational_order_parameter = std::numeric_limits<double>::quiet_NaN();
        local_entanglement = std::numeric_limits<double>::quiet_NaN();
        return Eigen::MatrixXd(0, 6);
    }

    // Extract local edges; quite stupid but what else can I do?
    Eigen::MatrixXd local_edges(local_edge_count, 6);
    Eigen::VectorXi local_edge_labels = Eigen::VectorXi::Zero(local_edge_count);
    Eigen::VectorXi local_edge_indices = Eigen::VectorXi::Zero(local_edge_count);
    int index = 0;
    for (int i = 0; i < all_edges.rows(); ++i) {
        if (local_edge_trues(i) == 1) {
            local_edge_indices(index) = i;
            local_edge_labels(index) = edge_labels(i);

            local_edges.row(index++) = all_edges.row(i);
        }
    }
    // Unique labels
    
    std::vector<int> unique_labels;
    unique_labels.reserve(local_edge_count); // Reserve memory to avoid multiple allocations
    for (int i = 0; i < local_edge_count; ++i) {
        
        int label = local_edge_labels(i);

        if (std::find(unique_labels.begin(), unique_labels.end(), label) == unique_labels.end()) {
            unique_labels.push_back(label);
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
    // Find argmax eigenvalue
    int max_eigenvalue_index = (eigenvalues.cwiseAbs().array() - max_eigenvalue).cwiseAbs().minCoeff();
    orientational_order_parameter = eigenvalues(max_eigenvalue_index);

    // Entanglement
    Eigen::MatrixXd entanglement_matrix(local_edge_count, local_edge_count);
    entanglement_matrix.setConstant(std::numeric_limits<double>::quiet_NaN());
    filamentFields::compute_edge_wise_entanglement(local_edges, local_edge_labels, entanglement_matrix);
    // ignore nan values and take absolute value and sum
    // double entanglement = entanglement_matrix.unaryExpr([](double x) { return std::abs(x); }).sum();
    // entanglement = entanglement_matrix.unaryExpr([](double x) {
    //     return std::isnan(x) ? 0.0 : std::abs(x);
    // }).sum();

    local_entanglement = entanglement_matrix.unaryExpr([](double x) -> double {
        return std::isnan(x) ? 0.0 : std::abs(x);
    }).sum();


    if (number_of_total_contacts > 0)
    {
        // sample the contact points
        Eigen::VectorXi contact_labels = Eigen::VectorXi::Zero(number_of_total_contacts);
        force_sum = 0;
        for (int i = 0; i < number_of_total_contacts; ++i) {
            Eigen::Vector3d contact_point = contact_array.row(i).segment<3>(0);
            if ((contact_point - query_point).norm() < R_omega) {
                contact_labels(i) = 1;
                // get contact_array's ith row from 4 to 6
                force_sum += contact_array.row(i).segment<3>(3).norm();
            }
        }
        // count the number of contacts
        number_of_local_contacts = contact_labels.sum();

        // compute the sum of forces
    }

    return local_edges;
}

Eigen::MatrixXd filamentFields::analyze_local_volume_from_precomputed(const Eigen::Vector3d& query_point, double R_omega, double rod_radius) {
    // make sure that the total_linking_matrix is computed
    if (is_precomputed == false) {
        precompute(R_omega);
    }

    // Sample edges locally
    Eigen::VectorXi local_edge_trues = sample_edges_locally(query_point, R_omega);
    number_of_labels = 0;
    // Count the number of local edges
    int local_edge_count = local_edge_trues.sum();
    if (local_edge_count == 0) {
        number_of_labels = 0;
        volume_fraction = std::numeric_limits<double>::quiet_NaN();
        orientational_order_parameter = std::numeric_limits<double>::quiet_NaN();
        local_entanglement = std::numeric_limits<double>::quiet_NaN();
        return Eigen::MatrixXd(0, 6);
    }

    // Extract local edges; quite stupid but what else can I do?
    Eigen::MatrixXd local_edges(local_edge_count, 6);
    Eigen::VectorXi local_edge_labels = Eigen::VectorXi::Zero(local_edge_count);
    Eigen::VectorXi local_edge_indices = Eigen::VectorXi::Zero(local_edge_count);
    int index = 0;
    for (int i = 0; i < all_edges.rows(); ++i) {
        if (local_edge_trues(i) == 1) {
            local_edge_indices(index) = i;
            local_edge_labels(index) = edge_labels(i);

            local_edges.row(index++) = all_edges.row(i);
        }
    }
    // Unique labels    
    std::vector<int> unique_labels;
    unique_labels.reserve(local_edge_count); // Reserve memory to avoid multiple allocations
    for (int i = 0; i < local_edge_count; ++i) {        
        int label = local_edge_labels(i);
        if (std::find(unique_labels.begin(), unique_labels.end(), label) == unique_labels.end()) {
            unique_labels.push_back(label);
            number_of_labels++;
        }
    }

    double edge_length_sum = 0.0;
    for (int i = 0; i < local_edge_count; ++i) {
        edge_length_sum += edge_lengths(local_edge_indices(i));
    }

    // volume fraction
    volume_fraction = (M_PI * rod_radius * rod_radius * edge_length_sum) / (4.0 / 3.0 * M_PI * R_omega * R_omega * R_omega);

    // Orientational order parameter
    Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
    for (int i = 0; i < local_edge_count; ++i) {
        Q += Q_tensors[local_edge_indices(i)];
    }
    Q /= local_edge_count;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(Q);
    Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
    // Find maximum absolute eigenvalue
    double max_eigenvalue = eigenvalues.cwiseAbs().maxCoeff();
    // Find argmax eigenvalue
    int max_eigenvalue_index = (eigenvalues.cwiseAbs().array() - max_eigenvalue).cwiseAbs().minCoeff();
    orientational_order_parameter = eigenvalues(max_eigenvalue_index);
    // local_Q_tensor = Q;
    // flatten the Q tensor
    local_Q_tensor << Q(0,0), Q(0,1), Q(0,2), Q(1,0), Q(1,1), Q(1,2), Q(2,0), Q(2,1), Q(2,2);

    // Entanglement
    Eigen::MatrixXd entanglement_matrix(local_edge_count, local_edge_count);
    entanglement_matrix.setConstant(std::numeric_limits<double>::quiet_NaN());

    for (int idx = 0; idx < local_edge_count; ++idx) {
        for (int jdx = idx+1; jdx < local_edge_count; ++jdx) {
            if (local_edge_labels(idx) == local_edge_labels(jdx)) {
                continue;
            }

            double debug1 = local_edge_indices(idx);
            double debug2 = local_edge_indices(jdx);
            double debug3 = local_edge_labels(idx);
            double debug4 = local_edge_labels(jdx);
            double e = total_linking_matrix(local_edge_indices(idx) , local_edge_indices(jdx));

            entanglement_matrix(idx,jdx) = total_linking_matrix(local_edge_indices(idx) , local_edge_indices(jdx));
        }
    }
    
    local_entanglement = entanglement_matrix.unaryExpr([](double x) -> double {
        return std::isnan(x) ? 0.0 : std::abs(x);
    }).sum();

    if (number_of_total_contacts > 0)
    {
        // sample the contact points
        Eigen::VectorXi contact_labels = Eigen::VectorXi::Zero(number_of_total_contacts);
        force_sum = 0;
        for (int i = 0; i < number_of_total_contacts; ++i) {
            Eigen::Vector3d contact_point = contact_array.row(i).segment<3>(0);
            if ((contact_point - query_point).norm() < R_omega) {
                contact_labels(i) = 1;
                // get contact_array's ith row from 4 to 6
                force_sum += contact_array.row(i).segment<3>(3).norm();
            }
        }
        // count the number of contacts
        number_of_local_contacts = contact_labels.sum();

        // compute the sum of forces
    }

    return local_edges;
}

// Eigen::MatrixXd filamentFields::analyze_local_volume_over_domain(const Eigen::MatrixX3d& query_points, double R_omega, double rod_radius) {
//     Eigen::MatrixXd results(query_points.rows(), 6);
//     for (int i = 0; i < query_points.rows(); ++i) {
//         Eigen::Vector3d query_point = query_points.row(i);
//         analyze_local_volume_from_precomputed(query_point, R_omega, rod_radius);
//         results.row(i) << number_of_labels, volume_fraction, orientational_order_parameter, local_entanglement, number_of_local_contacts, force_sum;
//     }
//     return results;
// }

Eigen::MatrixXd filamentFields::analyze_local_volume_over_domain(const Eigen::MatrixX3d& query_points, double R_omega, double rod_radius) {
    Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> results(query_points.rows(), 6);
    for (int i = 0; i < query_points.rows(); ++i) {
        Eigen::Vector3d query_point = query_points.row(i);
        analyze_local_volume_from_precomputed(query_point, R_omega, rod_radius);
        results.row(i) << number_of_labels, volume_fraction, orientational_order_parameter, local_entanglement, number_of_local_contacts, force_sum;
    }
    return results;
}


double filamentFields::_clip(double x, double lower, double upper) const {
    return std::min(std::max(x, lower), upper);
}

double filamentFields::compute_linking_number_for_edges(const Eigen::VectorXd& e_i, const Eigen::VectorXd& e_j) const {
    // Panagioutu and Kauffmand (Banchoff)
    // Eigen::Vector3d r_ij = e_i.segment<3>(0) - e_j.segment<3>(0);
    // Eigen::Vector3d r_ijj = e_i.segment<3>(0) - e_j.segment<3>(3);
    // Eigen::Vector3d r_iij = e_i.segment<3>(3) - e_j.segment<3>(0);
    // Eigen::Vector3d r_iijj = e_i.segment<3>(3) - e_j.segment<3>(3);

    // double tol = 1e-6;
    
    // Eigen::Vector3d n1 = r_ij.cross(r_ijj);
    // n1 /= (n1.norm() + tol);
    
    // Eigen::Vector3d n2 = r_ijj.cross(r_iijj);
    // n2 /= (n2.norm() + tol);
    
    // Eigen::Vector3d n3 = r_iijj.cross(r_iij);
    // n3 /= (n3.norm() + tol);
    
    // Eigen::Vector3d n4 = r_iij.cross(r_ij);
    // n4 /= (n4.norm() + tol);

    // return -1.0 / (4 * M_PI) * std::abs(
    //     std::asin(filamentFields::_clip(n1.dot(n2), -1.0 + tol, 1.0 - tol)) +
    //     std::asin(filamentFields::_clip(n2.dot(n3), -1.0 + tol, 1.0 - tol)) +
    //     std::asin(filamentFields::_clip(n3.dot(n4), -1.0 + tol, 1.0 - tol)) +
    //     std::asin(filamentFields::_clip(n4.dot(n1), -1.0 + tol, 1.0 - tol))
    // );

    // Qu and James (Arai)
    Eigen::Vector3d a = e_i.segment<3>(0) - e_j.segment<3>(0);
    Eigen::Vector3d b = e_i.segment<3>(0) - e_j.segment<3>(3);
    Eigen::Vector3d c = e_i.segment<3>(3) - e_j.segment<3>(3);
    Eigen::Vector3d d = e_i.segment<3>(3) - e_j.segment<3>(0);

    double lk_ij = INV_TWO_PI * (std::atan2( a.dot(b.cross(c)),(a.norm() * b.norm() * c.norm() + a.dot(b) * c.norm() + c.dot(a)*b.norm() + b.dot(c)*a.norm()) ) +
                    std::atan2( c.dot(d.cross(a)),(c.norm() * d.norm() * a.norm() + c.dot(d) * a.norm() + a.dot(c)*d.norm() + d.dot(a)*c.norm()) ));

    return lk_ij;
}



void filamentFields::compute_edge_wise_entanglement(const Eigen::MatrixXd& _all_edges, const Eigen::VectorXi& _edge_labels, Eigen::MatrixXd& entanglement_matrix) {
    int num_all_edges = _all_edges.rows();

    for (int idx = 0; idx < num_all_edges; ++idx) {
        const Eigen::VectorXd edge1 = _all_edges.row(idx);

        for (int jdx = idx + 1; jdx < num_all_edges; ++jdx) {
            if (_edge_labels(idx) == _edge_labels(jdx)) {
                continue;
            }
            const Eigen::VectorXd edge2 = _all_edges.row(jdx);
            // entanglement_matrix(idx, jdx) = filamentFields::compute_linking_number_for_edges(edge1, edge2);
            double lk = filamentFields::compute_linking_number_for_edges(edge1, edge2);
            entanglement_matrix(idx, jdx) = lk;
        }
    }

    // for (auto& pair : edge_pairs) {
    //     int idx = pair.first;
    //     int jdx = pair.second;
    //     if (_edge_labels(idx) == _edge_labels(jdx)) {
    //         continue;
    //     }
    //     const Eigen::VectorXd edge1 = _all_edges.row(idx);
    //     const Eigen::VectorXd edge2 = _all_edges.row(jdx);
    //     double lk = filamentFields::compute_linking_number_for_edges(edge1, edge2);
    //     entanglement_matrix(idx, jdx) = lk;
    // }
}




