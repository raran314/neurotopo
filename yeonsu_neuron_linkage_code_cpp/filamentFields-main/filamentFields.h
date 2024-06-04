#ifndef FILAMENT_FIELDS_H
#define FILAMENT_FIELDS_H

#include <cmath> // For M_PI and standard math functions
#include <vector>
#include <Eigen/Dense>

class filamentFields {
public:
    filamentFields(const std::vector<Eigen::MatrixXd>& filament_nodes_list);
    filamentFields(const std::vector<Eigen::MatrixXd>& filament_nodes_list, const Eigen::MatrixXd& contact_array);

    void update_filament_nodes_list(const std::vector<Eigen::MatrixXd>& _filament_nodes_list);
    void update_contact_array(const Eigen::MatrixXd& _contact_array);

    std::vector<Eigen::MatrixXd> return_filament_nodes_list() const { return filament_nodes_list; }
    std::vector<Eigen::MatrixXd> return_filament_edges_list() const { return filament_edges_list; }
    Eigen::MatrixXd return_all_nodes() const { return all_nodes; }
    Eigen::MatrixXd return_all_edges() const { return all_edges; }
    Eigen::VectorXi return_node_labels() const { return node_labels; }
    Eigen::MatrixXi return_edge_labels() const { return edge_labels; }
    Eigen::MatrixXd return_total_linking_matrix() const { return total_linking_matrix; }

    int return_number_of_labels() const { return number_of_labels; }
    double return_volume_fraction() const { return volume_fraction; }
    double return_orientational_order_parameter() const { return orientational_order_parameter; }
    Eigen::Matrix<double,9,1> return_local_Q_tensor() const { return local_Q_tensor; }
    double return_entanglement() const { return local_entanglement; }
    double return_total_entanglement() const { return total_entanglement; }
    int return_number_of_local_contacts() const { return number_of_local_contacts; }
    double return_force_sum() const { return force_sum; }

    // Eigen::MatrixXd return_entanglement_matrix() const { return entanglement_matrix; }

    Eigen::VectorXi sample_edges_locally(const Eigen::Vector3d& query_point, double R) const;

    Eigen::MatrixXd analyze_local_volume(const Eigen::Vector3d& query_point, double R_omega, double rod_radius);
    Eigen::MatrixXd analyze_local_volume_from_precomputed(const Eigen::Vector3d& query_point, double R_omega, double rod_radius);    
    Eigen::MatrixXd analyze_local_volume_over_domain(const Eigen::MatrixX3d& query_points, double R_omega, double rod_radius);

    double compute_linking_number_for_edges(const Eigen::VectorXd& e_i, const Eigen::VectorXd& e_j) const;
    void compute_edge_wise_entanglement(const Eigen::MatrixXd& _all_edges, const Eigen::VectorXi& _edge_labels, Eigen::MatrixXd& entanglement_matrix);

    void precompute(double R_omega);
    void compute_total_linking_matrix();
    void compute_all_Q_tensors();
    void compute_all_edge_lengths();
    

private:
    // these are local variables
    int number_of_labels;
    double volume_fraction;
    double orientational_order_parameter;
    double local_entanglement;
    double total_entanglement;
    Eigen::Matrix<double,9,1> local_Q_tensor;

    int number_of_local_contacts;
    double force_sum;
    

    std::vector<Eigen::MatrixXd> filament_nodes_list;
    std::vector<Eigen::MatrixXd> filament_edges_list;

    Eigen::MatrixXd contact_array;
    int number_of_total_contacts;

    Eigen::MatrixXd all_nodes;
    Eigen::MatrixXd all_edges;

    Eigen::VectorXi node_labels;
    Eigen::VectorXi edge_labels;

    std::vector< std::pair<int, int> > edge_pairs;

    Eigen::MatrixXd total_linking_matrix;
    std::vector<Eigen::MatrixXd> Q_tensors; // vs. whole Eigen matrix?
    Eigen::VectorXd edge_lengths; 

    void get_node_labels();
    void get_edge_labels();

    void get_all_nodes();
    void get_all_edges();

    void get_edge_pairs(double R_omega);

    double _clip(double x, double lower, double upper) const;

    bool is_precomputed = false;

};

#endif // FILAMENT_FIELDS_H
