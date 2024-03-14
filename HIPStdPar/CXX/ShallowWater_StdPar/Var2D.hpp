#include <cassert>

struct Range {
   const int beginX, endX;
   const int sizeX;
   Range(int beginX, int endX) : 
      beginX(beginX), endX(endX), sizeX(endX-beginX) {
         assert(beginX < endX);
         assert(sizeX !=0);
   }
};

struct Range2D {
   const int beginY, endY;
   const int beginX, endX;
   const int sizeY, sizeX;
   Range2D(int beginY, int endY, int beginX, int endX) : 
      beginY(beginY), endY(endY), beginX(beginX), endX(endX),
      sizeY(endY-beginY), sizeX(endX-beginX) {
         assert(beginY < endY);
         assert(beginX < endX);
         assert(sizeY !=0);
         assert(sizeX !=0);
   }
};

struct Var2D {
   int sizeY, sizeX;
   double *data;

   Var2D(int sizeY, int sizeX) : sizeY(sizeY), sizeX(sizeX) {
        data = (double *)aligned_alloc(128,sizeof(double) * sizeX * sizeY);
        //hipMallocManaged(&data,sizeof(double) * sizeX * sizeY);
   }

   double &operator()(int j, int i) const { return data[j*sizeX + i];}
   double *data_ptr() { return data; }
};

class range {
public:
    class iterator {
        friend class range;
    public:

        using difference_type = typename std::make_signed_t<int>;
        using value_type = int;
        using pointer = const int*;
        using reference = int;
        using iterator_category = std::random_access_iterator_tag;

        reference operator *() const { return i_; }
        iterator &operator ++() { ++i_; return *this; }
        iterator operator ++(int) { iterator copy(*this); ++i_; return copy; }

        iterator &operator --() { --i_; return *this; }
        iterator operator --(int) { iterator copy(*this); --i_; return copy; }

        iterator &operator +=(int by) { i_+=by; return *this; }

        value_type operator[](const difference_type &i) const { return i_ + i; }

    difference_type operator-(const iterator &it) const { return i_ - it.i_; }
    iterator operator+(const value_type v) const { return iterator(i_ + v); }

        bool operator ==(const iterator &other) const { return i_ == other.i_; }
        bool operator !=(const iterator &other) const { return i_ != other.i_; }
        bool operator < (const iterator &other) const { return i_ < other.i_; }

    protected:
        explicit iterator(int start) : i_ (start) {}

    private:
        int i_;
    };

    iterator begin() const { return begin_; }
    iterator end() const { return end_; }
    range(int begin, int end) : begin_(begin), end_(end) {}
private:
    iterator begin_;
    iterator end_;
};
