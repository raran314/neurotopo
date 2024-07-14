#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>

// Function to read CSV file into a vector of Eigen matrices
std::vector<Eigen::Matrix<double, 3, 1>> readCSVToEigen(const std::string& filename) {
    std::vector<Eigen::Matrix<double, 3, 1>> points;
    std::ifstream file(filename);
    std::string line;
    bool isFirstLine = true;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    while (std::getline(file, line)) {
        if (isFirstLine) {
            // Skip the first line (header)
            isFirstLine = false;
            continue;
        }

        std::stringstream ss(line);
        std::string item;
        std::vector<double> values;

        while (std::getline(ss, item, ',')) {
            values.push_back(std::stod(item));
        }

        if (values.size() == 3) {
            Eigen::Matrix<double, 3, 1> point;
            point << values[0], values[1], values[2];
            points.push_back(point);
        } else {
            std::cerr << "Invalid data in CSV file: " << line << std::endl;
            exit(1);
        }
    }

    file.close();
    return points;
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

// Function to calculate the writhe of a curve
double calculateWrithe(const std::vector<Eigen::Matrix<double, 3, 1>>& points) {
    int n = points.size();
    double writhe = 0.0;

    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n - 1; ++j) {
            Eigen::Vector3d r1 = points[i];
            Eigen::Vector3d r2 = points[i + 1];
            Eigen::Vector3d r3 = points[j];
            Eigen::Vector3d r4 = points[j + 1];

            Eigen::Vector3d r13 = r3 - r1;
            Eigen::Vector3d r24 = r4 - r2;
            Eigen::Vector3d r12 = r2 - r1;
            Eigen::Vector3d r34 = r4 - r3;

            Eigen::Vector3d cross12_34 = r12.cross(r34);
            Eigen::Vector3d cross13_24 = r13.cross(r24);

            double numerator = r13.dot(cross12_34);
            double denominator = std::pow(r13.norm() * r24.norm(), 1.5);

            writhe += numerator / denominator;
        }
    }

    writhe *= (1.0 / (4.0 * M_PI));
    return writhe;
}

// Function to calculate the sum of Euclidean distances between consecutive points
double calculateTotalEuclideanDistance(const std::vector<Eigen::Matrix<double, 3, 1>>& points) {
    double total_distance = 0.0;
    for (size_t i = 0; i < points.size() - 1; ++i) {
        total_distance += (points[i + 1] - points[i]).norm();
    }
    return total_distance;
}

int main() {
    std::string filename = "/Users/romyaran/Summer_2024/yeonsu_neuron_linkage_code_cpp/filamentFields-main/more_data/axonal_whorls/whorl_3/skel6.csv";
    std::vector<Eigen::Matrix<double, 3, 1>> points = readCSVToEigen(filename);

    double writhe = calculateWrithe(points);
    std::cout << "Writhe of the curve: " << writhe << std::endl;

    double total_distance = calculateTotalEuclideanDistance(points);
    std::cout << "Total Euclidean Distance: " << total_distance << std::endl;

    return 0;
}
