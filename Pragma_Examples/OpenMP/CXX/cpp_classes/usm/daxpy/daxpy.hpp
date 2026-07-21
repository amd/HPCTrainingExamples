#include <iostream>
#include <vector>

class daxpy {

private:
    double a_;          
    std::vector<double> x_; 
    std::vector<double> y_; 
    int N_;         
 
public:
    // constructor	
    daxpy(double a, int N) {
        a_ = a;
        N_ = N;
        x_.assign(N,0.0);
        y_.assign(N,0.0);
    }

    // destructor 
    ~daxpy() {
    }

    void setX(int index, double val) {
        x_[index]=val;
    }

    void setY(int index, double val) {
        y_[index]=val;
    }

    double getX(int index) {
        return x_[index];
    }

    double getY(int index) {
        return y_[index];
    }

    double getConst() const {
        return a_;
    }

    void setConst(double a) {
        a_ = a;
    }

    int getSize() const {
        return N_;
    }

    void setSize(int N) {
        N_ = N;
    }

    void apply();

    void printArrays() const {
        std::cout << "Array x: ";
        for (int i = 0; i < N_; ++i) {
            std::cout << x_[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Array y: ";
        for (int i = 0; i < N_; ++i) {
            std::cout << y_[i] << " ";
        }
        std::cout << std::endl;
    }

};

