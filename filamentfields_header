#ifndef FILAMENTFIELDS_H
#define FILAMENTFIELDS_H

#include <vector>
#include <armadillo>

class filamentFields {
public:
    filamentFields(const std::vector<arma::mat>& filament_nodes_list);
    
    arma::vec sample_edges_locally(const arma::vec& query_point, double R_omega) const;
    arma::mat analyzeLocalVolume(const arma::vec& query_point, double R_omega, double rod_radius);

private:
    void get_all_nodes();
    void get_all_edges();
    void get_node_labels();
    void get_edge_labels();
    double compute_linking_number_for_edges(const arma::rowvec& e_i, const arma::rowvec& e_j) const;
    void compute_edge_wise_entanglement(const arma::mat& _all_edges, const arma::vec& labels, arma::mat& entanglement_matrix) const;
    double _clip(double x, double lower, double upper) const;

    std::vector<arma::mat> filament_nodes_list;
    std::vector<arma::mat> filament_edges_list;
    arma::mat all_nodes;
    arma::mat all_edges;
    arma::vec node_labels;
    arma::vec edge_labels;
    int number_of_labels;
    double volume_fraction;
    double orientational_order_parameter;
    double entanglement;
};

#endif // FILAMENTFIELDS_H
