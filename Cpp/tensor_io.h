// tensor_io.h
// Utilitaires partages par TOUS les testbenchs du projet :
//   - load_tensor    : lit un fichier "shape\nvalues..." -> vecteur de floats
//   - load_matrix    : charge un fichier dans un tableau 2D de taille fixe (ROWS x COLS)
//   - load_vector    : charge un fichier dans un tableau 1D de taille fixe (SIZE)
//   - compare_tensors: compare deux vecteurs de floats avec une tolerance, affiche PASS/FAIL
#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>

inline std::vector<float> load_tensor(const std::string& path, std::vector<int>& shape_out) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERREUR : impossible d'ouvrir " << path << std::endl;
        std::exit(1);
    }

    std::string shape_line;
    std::getline(file, shape_line);
    std::istringstream shape_stream(shape_line);
    shape_out.clear();
    int d;
    while (shape_stream >> d) shape_out.push_back(d);

    std::vector<float> values;
    float v;
    while (file >> v) values.push_back(v);
    return values;
}

// Charge dans un tableau 2D [ROWS][COLS]. Ne verifie que le nombre total de valeurs
// (peu importe que le fichier Python ait une dimension batch=1 en plus, ex "1 16 32").
template<int ROWS, int COLS>
void load_matrix(const std::string& path, float out[ROWS][COLS]) {
    std::vector<int> shape;
    std::vector<float> values = load_tensor(path, shape);
    if ((int)values.size() != ROWS * COLS) {
        std::cerr << "ERREUR taille pour " << path << " : attendu " << ROWS * COLS
                   << " valeurs, trouve " << values.size() << std::endl;
        std::exit(1);
    }
    int idx = 0;
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            out[r][c] = values[idx++];
}

template<int SIZE>
void load_vector(const std::string& path, float out[SIZE]) {
    std::vector<int> shape;
    std::vector<float> values = load_tensor(path, shape);
    if ((int)values.size() != SIZE) {
        std::cerr << "ERREUR taille pour " << path << " : attendu " << SIZE
                   << " valeurs, trouve " << values.size() << std::endl;
        std::exit(1);
    }
    for (int i = 0; i < SIZE; i++) out[i] = values[i];
}

// Compare deux tenseurs aplatis, affiche PASS/FAIL + erreur max absolue.
inline bool compare_tensors(const std::vector<float>& computed,
                             const std::vector<float>& reference,
                             const std::string& label,
                             float tol = 1e-4f) {
    if (computed.size() != reference.size()) {
        std::cout << "[" << label << "] FAIL - tailles differentes ("
                   << computed.size() << " vs " << reference.size() << ")" << std::endl;
        return false;
    }
    float max_err = 0.0f;
    for (size_t i = 0; i < computed.size(); i++) {
        float err = std::fabs(computed[i] - reference[i]);
        if (err > max_err) max_err = err;
    }
    bool ok = max_err < tol;
    std::cout << "[" << label << "] " << (ok ? "PASS" : "FAIL")
               << " - erreur max absolue = " << max_err << std::endl;
    return ok;
}