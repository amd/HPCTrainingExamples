#include <iostream>
#include <string>
#include <vector>

class norm {

private:
    std::string type_;           
    std::vector<double> x_;         
    int N_; 
    double norm_;
 
public:
    // constructor	
    norm(std::string type, int N) {
        type_ = type;
        N_ = N;
        x_.assign(N, 0.0);
        norm_ = 0.0; 
    }

    // destructor 
    ~norm() {
    }

    void setX(int index, double val) {
        x_[index]=val;
    }

    double getX(int index) {
        return x_[index];
    }

    std::string getType() const {
        return type_;
    }

    int getSize() const {
        return N_;
    }

    double getNorm() const {
       return norm_;
    }

    void setSize(int N) {
        N_ = N;
    }

    void apply();

    void printNorm() const {
        std::cout << type_ << "Norm of array is : " << norm_ << std::endl; 
    }

};

