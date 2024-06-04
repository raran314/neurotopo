#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "filamentFields.h"
#include <fstream>
#include <string>
#include <sstream>

// Function to read CSV file and convert to Eigen matrices
std::vector<Eigen::MatrixXd> read_csv_to_eigen_matrices(const std::string& filename) {
    std::vector<Eigen::MatrixXd> eigen_matrices;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> values;
        int rows, cols;

        // Read shape information
        std::getline(ss, value, ',');
        rows = std::stoi(value);
        std::getline(ss, value, ',');
        cols = std::stoi(value);

        // Read matrix data
        while (std::getline(ss, value, ',')) {
            values.push_back(std::stod(value));
        }

        // Convert to Eigen matrix
        Eigen::MatrixXd matrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix(i, j) = values[i * cols + j];
            }
        }
        
        eigen_matrices.push_back(matrix);
    }

    return eigen_matrices;
}

int main() {
    std::vector<Eigen::MatrixXd> eigen_matrices = read_csv_to_eigen_matrices("arrays.csv");
    std::vector<Eigen::MatrixXd> filament_nodes_list;
    
    // Loop over the eigen_matrices vector and add each matrix to filament_nodes_list
    for (const auto& matrix : eigen_matrices) {
        filament_nodes_list.push_back(matrix);
    }

//    int num_rods = filament_nodes_list.size();
//     Seed the random number generator
//    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    // generate random rods
//    for (int i = 0; i < num_rods; i++) {
//        Eigen::MatrixXd nodes(10, 3);
//        for (int j = 0; j < 10; j++) {
//            nodes(j, 0) = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
//            nodes(j, 1) = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
//            nodes(j, 2) = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
//        }
//        filament_nodes_list.push_back(nodes);
//    }

    filamentFields filament(filament_nodes_list);
    Eigen::Vector3d query_point(0.0, 0.0, 0.0);
    double R_omega = 1;
    double rod_radius = 1;
//    filament.precompute(R_omega);
    
    // Uncomment these lines to use the filament analysis
    filament.analyze_local_volume_from_precomputed(query_point, R_omega, rod_radius);
    std::cout << "Number of labels: " << filament.return_number_of_labels() << std::endl;
    std::cout << "Volume fraction: " << filament.return_volume_fraction() << std::endl;
    std::cout << "Orientational order parameter: " << filament.return_orientational_order_parameter() << std::endl;
    std::cout << "Entanglement: " << filament.return_entanglement() << std::endl;
    
    return 0;
}
