#include <iostream>
#include "daxpy.hpp"
#include "norm.hpp"

class operations {

private:
    daxpy d_;
    norm n_;          
 
public:
    // constructor	
    operations(double a, int N, std::string type) : d_(a,N),n_(type,N){
    }

    // destructor 
    ~operations() {
    }

    void normSetX(int index, double val) {
        n_.setX(index,val);
    }

    void daxpySetX(int index, double val) {
        d_.setX(index,val);
    }

    void daxpySetY(int index, double val) {
        d_.setY(index,val);
    }

    void setBoth(int index, double val) {
        d_.setY(index,val);
        n_.setX(index,val);
    }

    double daxpyGetX(int index) {
        return d_.getX(index);
    }

    double daxpyGetY(int index) {
        return d_.getY(index);
    }

    double normGetX(int index) {
        return n_.getX(index);
    }

    double getNorm() const {
       n_.getNorm();
    }

    double getConst() const {
        return d_.getConst();
    }

    void setConst(double a) {
        d_.setConst(a);
    }

    int getSize() const {
        return d_.getSize();
    }

    void setSize(int N) {
        d_.setSize(N);
        n_.setSize(N);
    }

    void updateNorm() {
       n_.apply();
    }

    void apply();

    void printAll(){
       std::cout<<"------------ FROM DAXPY MEMBER "<< std::endl;
       d_.printArrays();
       std::cout<<"------------ FROM NORM MEMBER "<< std::endl;
       n_.printNorm();
    }

};

