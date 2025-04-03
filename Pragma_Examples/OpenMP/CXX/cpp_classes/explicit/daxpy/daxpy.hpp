#include <iostream>

class daxpy {

private:
    double a_;           
    double* x_;         
    double* y_;     
    int N_;         
 
public:
    // constructor	
    daxpy(double a, int N) {
        a_ = a;
        N_ = N;
        x_ = new double[N];
        y_ = new double[N];
        #pragma omp target enter data map(alloc: x_[0:N_],y_[0:N_], N_, a_)
    }

    // destructor 
    ~daxpy() {
        #pragma omp target exit data map(delete: x_[0:N_],y_[0:N_], N_, a_)
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
#pragma omp target update from(x_[0:N_],y_[0:N_])
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

