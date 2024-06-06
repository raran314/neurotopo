#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "filamentFields.h"
#include <fstream>
#include <string>
#include <sstream>

std::vector<Eigen::MatrixXd> read_csv_to_eigen_matrices(const std::string& filename) {
    std::vector<Eigen::MatrixXd> eigen_matrices;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return eigen_matrices;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> values;

        // Read shape information
        std::getline(ss, value, ',');
        int rows = std::stoi(value);
        std::getline(ss, value, ',');
        int cols = std::stoi(value);

        // Read matrix data
        while (std::getline(ss, value, ',')) {
            values.push_back(std::stod(value));
        }

        // Ensure correct number of values
        if (values.size() != rows * cols) {
            std::cerr << "Mismatch between expected and actual number of matrix elements" << std::endl;
            continue;
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

// Function to write Eigen matrices to CSV file
void write_eigen_matrices_to_csv(const std::vector<Eigen::MatrixXd>& matrices, const std::string& filename) {
    std::ofstream file(filename);

    for (const auto& matrix : matrices) {
        file << matrix.rows() << "," << matrix.cols();
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                file << "," << matrix(i, j);
            }
        }
        file << "\n";
    }
}

int main() {
    std::vector<Eigen::MatrixXd> eigen_matrices = read_csv_to_eigen_matrices("/Users/romyaran/Downloads/arrays.csv");
    std::vector<Eigen::MatrixXd> filament_nodes_list;
    
    // Example: Filling eigen_matrices with some dummy matrices for demonstration
//    eigen_matrices.push_back(Eigen::MatrixXd::Random(10, 3));
//    eigen_matrices.push_back(Eigen::MatrixXd::Random(10, 3));

    // Debug statement to verify the matrices are added
    std::cout << "Number of matrices in eigen_matrices: " << eigen_matrices.size() << std::endl;

//    if (eigen_matrices.empty()) {
//        std::cout << "eigen_matrices is empty. No matrices to process." << std::endl;
//    } else {
//        for (const auto& matrix : eigen_matrices) {
//            filament_nodes_list.push_back(matrix);
//            std::cout << "No need to store this string" << std::endl;
//            std::cout << "Matrix:\n" << matrix << "\n\n";
//            std::cout.flush(); // Ensure the output is immediately flushed to the console
//        }
//    }

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
    
    //std::cout << "No need to store this string";
    // Write the Eigen matrices to a new CSV file
    write_eigen_matrices_to_csv(eigen_matrices, "eigen_matrices.csv");
    
    filamentFields filament(eigen_matrices);
    
    Eigen::Vector3d query_point(10, 10, 10);
    double R_omega = 1;
    double rod_radius = 1;
//    filament.precompute(R_omega);
    
    // Uncomment these lines to use the filament analysis
    //filament.analyze_local_volume_from_precomputed(query_point, R_omega, rod_radius);
    std::cout << "Number of labels: " << filament.return_number_of_labels() << std::endl;
//    std::cout << "Number of labels: " << filament.return_label_list() << std::endl;
    std::cout << "Volume fraction: " << filament.return_volume_fraction() << std::endl;
    std::cout << "Orientational order parameter: " << filament.return_orientational_order_parameter() << std::endl;
    std::cout << "Entanglement: " << filament.return_entanglement() << std::endl;
    std::cout << "Linking matrix: " << filament.return_total_linking_matrix() << std::endl;
    return 0;
}
