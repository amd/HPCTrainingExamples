#include <iostream>

class daxpy {

private:
    double a_;           
    double* x_;         
    double* y_;     
    int N_;         
 
#pragma omp declare mapper (class daxpy d) map(tofrom: d, d.x_[:d.N_], d.y_[:d.N_])

public:
    // constructor	
    daxpy(double a, int N) {
        a_ = a;
        N_ = N;

        x_ = new double[N];
        y_ = new double[N];
    }

    // destructor 
    ~daxpy() {
        delete[] x_;
        delete[] y_;
    }

#pragma omp begin declare target
    void setX(int index, double val) {
        x_[index]=val;
    }
#pragma omp end declare target

#pragma omp begin declare target
    void setY(int index, double val) {
        y_[index]=val;
    }
#pragma omp end declare target

#pragma omp begin declare target
    double getX(int index) {
        return x_[index];
    }
#pragma omp end declare target

#pragma omp begin declare target
    double getY(int index) {
        return y_[index];
    }
#pragma omp end declare target

#pragma omp begin declare target
    double getConst() const {
        return a_;
    }
#pragma omp end declare target

#pragma omp begin declare target
    void setConst(double a) {
        a_ = a;
    }
#pragma omp end declare target

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

